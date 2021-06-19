#!/usr/bin/env python3
import math

import numpy as np
import argparse
import configparser

import tensorflow as tf

keras = tf.keras


class SignalGenerationLayer3D(keras.layers.Layer):
    """
    Encapsulate all the signal generation code into a Keras layer
    """

    def __init__(self, system_parameters, full_model, include_blood):
        """
        Create a signal generation layer based on the forward equations from OEF/DBV
        :param system_parameters: A dictionary contain the model system parameters
        :param full_model: boolean, do we use the full or log-linear model
        :param include_blood: boolean: do we include the contribution of blood
        """

        # TODO: Are any of these parameters something we might want to infer from data?
        self._gamma = float(system_parameters['gamma'])
        self._b0 = float(system_parameters['b0'])
        self._dchi = float(system_parameters['dchi'])
        self._hct = float(system_parameters['hct'])
        self._te = float(system_parameters['te'])
        self._r2t = float(system_parameters['r2t'])
        self._taus = tf.range(float(system_parameters['tau_start']), float(system_parameters['tau_end']),
                              float(system_parameters['tau_step']), dtype=tf.float32)

        self._tr = float(system_parameters['tr'])
        self._ti = float(system_parameters['ti'])
        self._t1b = float(system_parameters['t1b'])

        self._simulate_noise = system_parameters['simulate_noise'] == 'True'
        self._weighted_noise = system_parameters['tau_weighted'] == 'True'
        self._snr = int(system_parameters['snr'])

        self._full_model = full_model
        self._include_blood = include_blood

        super().__init__()

    def call(self, input, *args, **kwargs):
        """
        Override the base class call method. This calculate the predicted signal (without added noise)
        given the input OEF/DBV
        :param inputs: a tensor of any shape, where the final dimension has size 2 to represent the OEF/DBV
        :return: The predicted signal
        """
        assert input.shape[-1] == 2, 'Input should have 2 elements in last dimension, OEF and DBV'
        # Store the original shape, ignoring the last two dimensions
        original_shape = input.shape[:-1]
        # Flatten the inputs except the last axis
        reshaped_input = tf.reshape(input, (-1, 2))
        # Assume oef and dbv are the only elements of the last axis
        oef, dbv = tf.split(reshaped_input, 2, axis=-1)

        tissue_signal = self.calc_tissue(oef, dbv)
        blood_signal = tf.zeros_like(tissue_signal)

        if self._include_blood:
            nb = 0.775
            m_bld = 1 - (2 - tf.math.exp(- (self._tr - self._ti) / self._t1b)) * tf.math.exp(-self._ti / self._t1b)
            vb = dbv

            blood_weight = m_bld * nb * vb
            blood_signal = self.calc_blood(oef)
        else:
            blood_weight = dbv

        tissue_weight = 1 - blood_weight

        signal = tissue_weight * tissue_signal + blood_weight * blood_signal

        # Normalise the data based on where tau = 0 to remove arbitrary scaling and take the log
        signal = tf.math.log(signal/tf.expand_dims(signal[:, tf.where(self._taus == 0)[0][0]], 1))

        if self._simulate_noise:
            if self._weighted_noise:
                noise_weights = tf.math.sqrt(2*tf.range(1, self._taus.shape[0]+1)/self._taus.shape[0])  # calculate weightings for noise varying over tau
                noise_weights = tf.cast(tf.expand_dims(noise_weights, 0), tf.float32)
            else:
                noise_weights = 1

            stdd = abs(tf.math.reduce_mean(signal, axis=1)) / self._snr  # calculate standard deviation for each signal
            stdd = tf.repeat(tf.expand_dims(stdd, 1), 11, axis=1)
            noise = tf.random.normal(signal.shape, tf.zeros(signal.shape), stdd, tf.float32)

            signal += noise * noise_weights  # add noise to each signal weighted on tau values

        # The predicted signal should have the original shape with len(self.taus)
        # TODO: Reshaping fixed in 3d version, needs to be fixed again in normal (should reshape after applying noise)
        signal = tf.reshape(signal, original_shape + (len(self._taus, )))

        return signal

    def calc_tissue(self, oef, dbv):
        """
        :param oef: A tensor containing the oef value of each parameter pair
        :param dbv: A tensor containing the dbv value of each parameter pair
        :return: The signal contribution from brain tissue
        """
        def compose(signal_idx):
            """
            :param signal_idx: The index for signal to calculate
            :return: The signal for the given index calculated using the full model
            """
            return tf.math.exp(-dbv[signal_idx] * integral((2 + int_parts) * tf.math.sqrt(1 - int_parts) * (
                    1 - tf.math.special.bessel_j0(1.5 * (tf.expand_dims(self._taus, 1) * dw[signal_idx]) * int_parts)) / (int_parts ** 2),
                                                           int_parts) / 3) * tf.math.exp(-self._te * self._r2t)

        def integral(y, x):
            """
            :param y: The y values for the sections between integral limits
            :param x: The x values of each section
            :return: The riemann integral used to calculate a full model signal
            """
            dx = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
            return tf.reduce_sum(tf.where(tf.math.is_nan(y[:, :-1]), tf.zeros_like(y[:, :-1]), y[:, :-1]), axis=1) * dx

        dw = (4 / 3) * math.pi * self._gamma * self._b0 * self._dchi * self._hct * oef
        tc = 1 / dw
        r2p = dw * dbv

        if self._full_model:
            a = tf.constant(0, dtype=tf.float32)  # lower limit for integration
            b = tf.constant(1, dtype=tf.float32)  # upper limit for integration
            int_parts = tf.linspace(a, b, 2 ** 5 + 1)  # riemann integral uses sum of many x values within limit to make estimate

            signals = tf.vectorized_map(compose, tf.range(dw.shape[0]))
        else:
            # Calculate the signals in both regimes and then sum and multiply by their validity. Although
            # this seems wasteful, but it's much easier to parallelise

            taus_under_tc = abs(self._taus) < tc
            taus_over_tc = taus_under_tc == False

            taus_under_tc = tf.cast(taus_under_tc, tf.float32)
            taus_over_tc = tf.cast(taus_over_tc, tf.float32)

            s = tf.exp(-self._r2t * self._te) * tf.exp(- (0.3 * (r2p * self._taus) ** 2) / dbv)
            s2 = tf.exp(-self._r2t * self._te) * tf.exp(dbv - (r2p * tf.abs(self._taus)))

            signals = s * taus_under_tc + s2 * taus_over_tc

        return signals

    def calc_blood(self, oef):
        """
        :param oef: A tensor containing the the oef values from each parameter pair
        :return: The signal contribution from venous blood
        """
        r2b = 4.5 + 16.4 * self._hct + (165.2 * self._hct + 55.7) * oef ** 2
        td = 0.0045067
        g0 = (4 / 45) * self._hct * (1 - self._hct) * ((self._dchi * self._b0) ** 2)

        signals = tf.math.exp(-r2b * self._te) * tf.math.exp(- (0.5 * (self._gamma ** 2) * g0 * (td ** 2)) *
                                                             ((self._te / td) + tf.math.sqrt(
                                                                 0.25 + (self._te / td)) + 1.5 -
                                                              (2 * tf.math.sqrt(
                                                                  0.25 + (((self._te + self._taus) ** 2) / td))) -
                                                              (2 * tf.math.sqrt(
                                                                  0.25 + (((self._te - self._taus) ** 2) / td)))))

        return signals


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']

    parser = argparse.ArgumentParser(description='Generate ASE qBOLD signals')

    parser.add_argument('-f',
                        required=True,
                        help='should the tissue contribution be calculated with the full model')
    parser.add_argument('-b',
                        required=True,
                        help='should the blood contribution be included')

    args = parser.parse_args()

    if args.f not in ['True', 'False'] or args.b not in ['True', 'False']:
        raise ValueError('Arguments must be a valid boolean')

    sig_layer = SignalGenerationLayer3D(params, args.f, args.b)

    shape = (68, 68, 68)
    radius = 20
    position = (34, 34, 34)

    semisizes = (radius,) * 3

    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]

    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += (x_i / semisize) ** 2

    sphere = (arr <= 1.0).astype(int)

    oefs = tf.random.normal(shape, 0.3, 0.054, tf.float32) * sphere
    dbvs = tf.random.normal(shape, 0.04, 0.019, tf.float32) * sphere

    train_y = tf.stack([oefs, dbvs], axis=3)
    train_y = tf.where(tf.equal(train_y, 0), tf.fill(train_y.shape, float('NaN')), train_y)
    train_x = sig_layer(train_y)

    train = tf.concat([train_x, train_y], -1)
    train = tf.random.shuffle(train)
    train_x = train[:, :, :, :11]
    train_y = train[:, :, :, 11:]

    np.save('signals', train_x)
    np.save('params', train_y)
