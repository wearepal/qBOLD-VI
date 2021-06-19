#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse
import configparser


def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv3D(18, kernel_size=(1, 1, 1), activation='relu'))
    for i in range(2):
        model.add(keras.layers.Conv3D(18, kernel_size=(1, 1, 1), activation='relu'))
    model.add(keras.layers.Dense(4))
    return model


def loss_fn(y_true, y_pred):
    oef_mean = y_pred[:, :, :, :, 0]
    oef_log_std = y_pred[:, :, :, :, 1]
    dbv_mean = y_pred[:, :, :, :, 2]
    dbv_log_std = y_pred[:, :, :, :, 3]
    oef_nll = -(-oef_log_std - (1 / 2) * ((y_true[:, :, :, :, 0] - oef_mean) / tf.exp(oef_log_std)) ** 2)
    dbv_nll = -(-dbv_log_std - (1 / 2) * ((y_true[:, :, :, :, 1] - dbv_mean) / tf.exp(dbv_log_std)) ** 2)

    keras.backend.print_tensor(oef_nll)

    nll = tf.add(oef_nll, dbv_nll)
    keras.backend.print_tensor(tf.reduce_mean(tf.boolean_mask(nll, not tf.math.is_nan(nll))))

    return tf.reduce_mean(tf.boolean_mask(nll, not tf.math.is_nan(nll)))


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']

    parser = argparse.ArgumentParser(description='Train neural network for parameter estimation')

    parser.add_argument('-s',
                        required=True,
                        help='path to signals file')
    parser.add_argument('-p',
                        required=True,
                        help='path to parameters file')

    args = parser.parse_args()

    x = np.load(args.s)
    y = np.load(args.p)
    crop_size = 20

    train_x = np.zeros((x.shape[0], crop_size, crop_size, crop_size, x.shape[-1]))
    train_y = np.zeros((y.shape[0], crop_size, crop_size, crop_size, y.shape[-1]))

    for i in range(x.shape[0]):
        # Generate the starting point for croppings
        crop = tf.random.uniform([3], 0, x.shape[1] - crop_size,
                                 dtype=tf.int32)  # TODO: this will break when dimensions not equal
        cropper = keras.layers.Cropping3D(
            cropping=tuple((crop[i], x.shape[1] - (crop[i] + crop_size)) for i in range(3)))  # Produces crop_size shaped croppings
        train_x[i] = cropper(tf.expand_dims(x[i], 0))
        train_y[i] = cropper(tf.expand_dims(y[i], 0))

    model = create_model()
    optimiser = tf.keras.optimizers.Adam()

    model.compile(optimiser, loss=loss_fn)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    mc = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', verbose=1)

    model.fit(train_x, train_y, epochs=500, callbacks=[es, mc], validation_split=0.2, batch_size=8)
