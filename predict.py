#!/usr/bin/env python3

import tensorflow as tf

import numpy as np
import argparse
import configparser


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']

    parser = argparse.ArgumentParser(description='Estimate parameters from ASE qBOLD signals')

    parser.add_argument('-s',
                        required=True,
                        help='path to signals file')

    args = parser.parse_args()

    x = np.genfromtxt(args.s, delimiter=',')

    model = tf.keras.models.load_model('model.h5', compile=False)

    predictions = model.predict(x)

    predictions[:, 1] = tf.math.exp(predictions[:, 1])
    predictions[:, 3] = tf.math.exp(predictions[:, 3])

    np.savetxt('predictions.csv', predictions, delimiter=',')
