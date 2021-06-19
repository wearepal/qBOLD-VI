#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

import numpy as np
import argparse
import configparser


def loss_fn(y_true, y_pred):
    oef_mean = y_pred[:, 0]
    oef_log_std = y_pred[:, 1]
    dbv_mean = y_pred[:, 2]
    dbv_log_std = y_pred[:, 3]
    oef_nll = -(-oef_log_std - (1/2)*((y_true[:, 0]-oef_mean)/tf.exp(oef_log_std))**2)
    dbv_nll = -(-dbv_log_std - (1/2)*((y_true[:, 1]-dbv_mean)/tf.exp(dbv_log_std))**2)

    nll = tf.add(oef_nll, dbv_nll)

    return tf.reduce_mean(nll)


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']

    parser = argparse.ArgumentParser(description='Estimate parameters from ASE qBOLD signals')

    parser.add_argument('-s',
                        required=True,
                        help='path to signals file')

    args = parser.parse_args()

    x = np.load(args.s)

    trained_model = keras.models.load_model('model.h5', compile=False)
    weights = trained_model.trainable_variables

    model = keras.models.Sequential()
    model.add(keras.layers.Conv3D(18,
                                  kernel_size=(1, 1, 1),
                                  activation='relu',
                                  kernel_initializer=tf.keras.initializers.constant(trained_model.layers[0].get_weights()[0]),
                                  bias_initializer=tf.keras.initializers.constant(trained_model.layers[0].get_weights()[1])))
    for i in range(2):
        model.add(keras.layers.Conv3D(18,
                                      kernel_size=(1, 1, 1),
                                      activation='relu',
                                      kernel_initializer=tf.keras.initializers.constant(trained_model.layers[i+1].get_weights()[0]),
                                      bias_initializer=tf.keras.initializers.constant(trained_model.layers[i+1].get_weights()[1])))
    model.add(keras.layers.Dense(4,
                                 kernel_initializer=tf.keras.initializers.constant(trained_model.layers[3].get_weights()[0]),
                                 bias_initializer=tf.keras.initializers.constant(trained_model.layers[3].get_weights()[1])))

    optimiser = tf.keras.optimizers.Adam()
    model.compile(optimiser, loss=loss_fn)

    x = tf.expand_dims(x, 0)

    predictions = model.predict(x)

    predictions[-1][1] = tf.math.exp(predictions[-1][1])
    predictions[-1][3] = tf.math.exp(predictions[-1][3])

    np.save('predictions', predictions)
