#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from signals import SignalGenerationLayer

import numpy as np
import argparse
import configparser


def create_model(use_conv=False):
    model = keras.models.Sequential()
    if use_conv:
        model.add(keras.layers.InputLayer(input_shape=(None, None, None, 11), ragged=False))
    else:
        model.add(keras.layers.InputLayer(input_shape=(11,)))

    def create_layer(no_units, activation='relu'):
        if use_conv:
            return keras.layers.Conv3D(no_units, kernel_size=(1, 1, 1), activation=activation)
        else:
            return keras.layers.Dense(no_units, activation=activation)

    for i in range(3):
        model.add(create_layer(18))

    # Removed sigmoid from output
    model.add(create_layer(4, None))
    return model


def forward_transform(logit):
    return tf.nn.sigmoid(logit)


def backwards_transform(signal):
    return tf.math.log(signal/(1.0-signal))


def loss_fn(y_true, y_pred):
    # Reshape the data such that we can work with either volumes or single voxels
    y_true = tf.reshape(y_true, (-1, 2))
    y_true = backwards_transform(y_true)
    y_pred = tf.reshape(y_pred, (-1, 4))

    oef_mean = y_pred[:, 0]
    oef_log_std = y_pred[:, 1]
    dbv_mean = y_pred[:, 2]
    dbv_log_std = y_pred[:, 3]
    oef_nll = -(-oef_log_std - (1.0 / 2.0) * ((y_true[:, 0] - oef_mean) / tf.exp(oef_log_std)) ** 2)
    dbv_nll = -(-dbv_log_std - (1.0 / 2.0) * ((y_true[:, 1] - dbv_mean) / tf.exp(dbv_log_std)) ** 2)

    nll = tf.add(oef_nll, dbv_nll)

    return tf.reduce_mean(nll)


class ReparamTrickLayer(keras.layers.Layer):
    def call(self, input, *args, **kwargs):
        oef_sample = input[:, :, :, :, 0] + tf.random.normal(tf.shape(input[:, :, :, :, 0])) * tf.exp(
            input[:, :, :, :, 1])
        dbv_sample = input[:, :, :, :, 2] + tf.random.normal(tf.shape(input[:, :, :, :, 0])) * tf.exp(
            input[:, :, :, :, 3])

        samples = tf.stack([oef_sample, dbv_sample], -1)
        samples = forward_transform(samples)
        samples = tf.clip_by_value(samples, clip_value_min=1e-3, clip_value_max=0.99)
        return samples


def fine_tune_loss_fn(y_true, y_pred):
    mask = y_true[:, :, :, :, -1]
    residual = y_true[:, :, :, :, :-1] - y_pred
    residual = tf.reduce_sum(tf.abs(residual), -1)
    residual = tf.where(mask > 0, residual, tf.zeros_like(residual))

    # keras.backend.print_tensor(residual)

    return tf.reduce_sum(residual) / tf.reduce_sum(mask)


def prepare_dataset(real_data):
    real_dataset = tf.data.Dataset.from_tensor_slices(real_data)

    def map_func(data):
        data_shape = data.shape.as_list()
        new_shape = data_shape[0:2] + [-1, ]
        data = tf.reshape(data, new_shape)
        data = tf.image.random_crop(value=data, size=(20, 20, data.shape[-1]))
        data = tf.reshape(data, [20, 20] + data_shape[-2:])
        mask = data[:, :, :, -1:]
        # Normalise within the mask
        data = tf.where(data[:, :, :, -1:] > 0, tf.math.log(data[:, :, :, :-1] / data[:, :, :, 2:3]),
                        tf.zeros_like(data[:, :, :, :-1]))
        # concat the mask
        data = tf.concat([data, mask], -1)
        return data[:, :, :, :-1], data

    real_dataset = real_dataset.map(map_func)
    real_dataset = real_dataset.batch(6, drop_remainder=True)
    real_dataset = real_dataset.repeat(-1)
    return real_dataset

def save_predictions(model, data, filename):
    import nibabel as nib

    data = tf.where(data[:, :, :, -1:] > 0, tf.math.log(data[:, :, :, :-1] / data[:, :, :, 2:3]),
             tf.zeros_like(data[:, :, :, :-1]))
    predictions = forward_transform(model.predict(data[:, :, :, :, :-1]))
    images = np.split(predictions[:,:,:,:,:], data.shape[0], axis=0)
    images = np.squeeze(np.concatenate(images, axis=-1),0)
    affine = np.eye(4)
    array_img = nib.Nifti1Image(images, affine)
    nib.save(array_img, filename)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']

    parser = argparse.ArgumentParser(description='Train neural network for parameter estimation')

    parser.add_argument('-f', default='synthetic_data.npz', help='path to synthetic data file')

    args = parser.parse_args()

    data_file = np.load(args.f)
    x = data_file['x']
    y = data_file['y']

    train_conv = True
    # If we're building a convolutional model, reshape the synthetic data to look like single voxel images
    if train_conv:
        x = np.reshape(x, (-1, 1, 1, 1, 11))
        y = np.reshape(y, (-1, 1, 1, 1, 2))

    optimiser = tf.keras.optimizers.Adam()
    model = create_model(train_conv)
    model.compile(optimiser, loss=loss_fn)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    mc = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1)

    model.fit(x, y, epochs=30, callbacks=[es, mc], validation_split=0.2, batch_size=8)

    # Load real data for fine-tuning
    real_data = np.load('/Users/is321/Documents/Data/qBold/hyperv_data/hyperv_ase.npy')
    real_dataset = prepare_dataset(real_data)

    valid_data = np.load('/Users/is321/Documents/Data/qBold/hyperv_data/baseline_ase.npy')
    print(valid_data.shape)
    valid_dataset = prepare_dataset(valid_data)

    save_predictions(model, valid_data, 'after_pt')

    full_optimiser = tf.keras.optimizers.Adam(learning_rate=1e-4)
    full_model = keras.Sequential()
    full_model.add(keras.layers.InputLayer((20, 20, 8, 11)))
    full_model.add(model)
    full_model.add(ReparamTrickLayer())
    params['simulate_noise'] = 'False'
    full_model.add(SignalGenerationLayer(params, False, True))

    full_model.compile(full_optimiser, loss=fine_tune_loss_fn)
    full_model.fit(real_dataset, validation_data=valid_dataset, steps_per_epoch=100, epochs=10, validation_steps=1)
    save_predictions(model, valid_data, 'after_ft')



