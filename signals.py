#!/usr/bin/env python3
import math

import numpy as np
import argparse
import configparser

import tensorflow as tf

keras = tf.keras


class SignalGenerationLayer(keras.layers.Layer):
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
            # Spin densities
            nb = 0.775
            # compartment steady-state magnetization adapted from Cherukara et al code
            # What paper does this come from?
            m_bld = 1 - (2 - tf.math.exp(- (self._tr - self._ti) / self._t1b)) * tf.math.exp(-self._ti / self._t1b)

            blood_weight = m_bld * nb * dbv
            blood_signal = self.calc_blood(oef)
        else:
            blood_weight = dbv

        tissue_weight = 1 - blood_weight

        signal = tissue_weight * tissue_signal + blood_weight * blood_signal

        if self._simulate_noise:
            # Normalised SNRs are given from real data, and calculated with respect to the tau=0 image
            norm_snr = np.array([0.985, 1.00, 1.01, 1., 0.97, 0.95, 0.93, 0.90, 0.86, 0.83, 0.79], dtype=np.float32)
            # The actual SNR varies between 60-120, but I've increased the range for more diversity
            snr = tf.random.uniform((signal.shape[0], 1), 5, 120) * tf.reshape(norm_snr, (1, 11))
            # Calculate the mean signal for each tau value and divie by the snr to get the std-dev
            std_dev = tf.reduce_mean(signal, 0, keepdims=True) / snr
            # Add noise at the correct level
            signal = signal + tf.random.normal(signal.shape) * std_dev

        """
        # Normalise the data based on where tau = 0 to remove arbitrary scaling and take the log
        tau_zero_data = signal[:, tf.where(self._taus == 0)[0][0]]
        signal = tf.math.log(signal/tf.expand_dims(tau_zero_data, 1))
        """

        # The predicted signal should have the original shape with len(self.taus)
        new_shape = original_shape.as_list() + [len(self._taus)]
        # Set the first value of new_shape to -1 (in case we have an unknown batch size)
        new_shape[0] = -1
        signal = tf.reshape(signal, new_shape)

        return signal

    def calc_tissue(self, oef, dbv):
        """
        :param oef: A tensor containing the oef value of each parameter pair
        :param dbv: A tensor containing the dbv value of each parameter pair
        :return: The signal contribution from brain tissue
        """

        def compose(arg):
            """
            :param signal_idx: The index for signal to calculate
            :return: The signal for the given index calculated using the full model
            """
            dbv_i, dw_i = arg
            # lower limit for integration, although it's defined between 0 and 1, 0 gives nans because of divide by 0
            # in integrand....
            a = tf.constant(1e-5, dtype=tf.float32)
            b = tf.constant(1, dtype=tf.float32)  # upper limit for integration
            int_parts = tf.linspace(a, b, 2 ** 8 + 1)

            return tf.math.exp(-dbv_i * integral((2 + int_parts) * tf.math.sqrt(1 - int_parts) * (
                    1.0 - tf.math.special.bessel_j0(1.5 * (tf.expand_dims(self._taus, 1) * dw_i) * int_parts))
                                                 / (3.0 * tf.square(int_parts)), int_parts)) \
                   * tf.math.exp(-self._te * self._r2t)

        def integral(y, x):
            """
            :param y: The y values for the sections between integral limits
            :param x: The x values of each section
            :return: The integral calculated using Simpson's rule
            """
            y_a = y[:, 0:-2:2]
            y_b = y[:, 2::2]
            y_m = y[:, 1:-1:2]
            h = (x[2] - x[0]) / 2.0
            integrals = (y_a + y_b + 4.0 * y_m) * (h / 3.0)
            return tf.reduce_sum(integrals, -1)

        dw = (4.0 / 3.0) * math.pi * self._gamma * self._b0 * self._dchi * self._hct * oef
        tc = 1.0 / dw
        r2p = dw * dbv

        if self._full_model:
            signals = tf.vectorized_map(compose, (dbv, dw))
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
        # R2b taken from Cherukara code - where is this derived from?
        r2b = 4.5 + 16.4 * self._hct + (165.2 * self._hct + 55.7) * oef ** 2
        td = 0.0045067
        # Why is the 4pi missing from the squared term here? Missing in cherukara code as well.
        g0 = (4 / 45) * self._hct * (1 - self._hct) * ((self._dchi * self._b0 * oef) ** 2)

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

    if False:
        params['simulate_noise'] = 'False'
        inp = tf.convert_to_tensor([[[[[0.4, 0.12]]]]])
        with tf.GradientTape() as tape:
            tape.watch(inp)
            o = SignalGenerationLayer(params, True, True)(inp)
            tf.keras.backend.print_tensor(o)
        print(tape.gradient(o, inp))

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

    sig_layer = SignalGenerationLayer(params, args.f, args.b)

    oefs = tf.random.uniform((int(params['sample_size']),), minval=float(params['oef_start']),
                             maxval=float(params['oef_end']))
    dbvs = tf.random.uniform((int(params['sample_size']),), minval=float(params['dbv_start']),
                             maxval=float(params['dbv_end']))
    xx, yy = tf.meshgrid(oefs, dbvs, indexing='ij')
    train_y = tf.stack([tf.reshape(xx, [-1]), tf.reshape(yy, [-1])], axis=1)

    # Remove any ordering from the data
    train_y = tf.random.shuffle(train_y)
    train_x_list = []
    # break into chunks to avoid running out of memory
    for i in range(10):
        chunk_size = train_y.shape[0] // 10
        y_subset = train_y[i * chunk_size:(i + 1) * chunk_size]
        train_x_list.append(sig_layer(y_subset))

    train_x = tf.concat(train_x_list, axis=0)

    np.savez('synthetic_data', x=train_x, y=train_y)
