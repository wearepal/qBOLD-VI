#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse
import configparser


def create_model(use_conv=False):
    model = keras.models.Sequential()
    if use_conv:
        model.add(keras.layers.InputLayer(input_shape=(None, None, None, 11), ragged=False))
    else:
        model.add(keras.layers.InputLayer(input_shape=(11,)))

    def create_layer(no_units):
        if use_conv:
            return keras.layers.Conv3D(no_units, kernel_size=(1, 1, 1), activation='relu')
        else:
            return keras.layers.Dense(no_units, activation='relu')

    for i in range(3):
        model.add(create_layer(18))

    # Removed sigmoid from output
    model.add(keras.layers.Dense(4))
    return model


def loss_fn(y_true, y_pred):
    # Reshape the data such that we can work with either volumes or single voxels
    y_true = tf.reshape(y_true, (-1, 2))
    y_pred = tf.reshape(y_pred, (-1, 4))

    oef_mean = y_pred[:, 0]
    oef_log_std = y_pred[:, 1]
    dbv_mean = y_pred[:, 2]
    dbv_log_std = y_pred[:, 3]
    oef_nll = -(-oef_log_std - (1 / 2) * ((y_true[:, 0] - oef_mean) / tf.exp(oef_log_std)) ** 2)
    dbv_nll = -(-dbv_log_std - (1 / 2) * ((y_true[:, 1] - dbv_mean) / tf.exp(dbv_log_std)) ** 2)

    nll = tf.add(oef_nll, dbv_nll)

    return tf.reduce_mean(nll)


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

    model = create_model(train_conv)
    optimiser = tf.keras.optimizers.Adam()

    model.compile(optimiser, loss=loss_fn)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    mc = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1)

    model.fit(x, y, epochs=500, callbacks=[es, mc], validation_split=0.2, batch_size=8)
