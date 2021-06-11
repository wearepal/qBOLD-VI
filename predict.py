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

    model = tf.keras.models.load_model('model.h5')

    predictions = model.predict(x)

    np.savetxt('predictions.csv', predictions, delimiter=',')
