#!/usr/bin/env python3
import argparse
import configparser
import math

import numpy as np
import tensorflow as tf

keras = tf.keras


class SignalGenerationLayer(keras.layers.Layer):
    """
    Encapsulate all the signal generation code into a Keras layer
    """

    def __init__(self, system_parameters, full_model, include_blood, misaligned_prob=0.0, variable_hct=False):
        """
        Create a signal generation layer based on the forward equations from OEF/DBV
        :param system_parameters: A dictionary contain the model system parameters
        :param full_model: boolean, do we use the full or log-linear model
        :param include_blood: boolean: do we include the contribution of blood
        :param misaligned_prob: float (0-1.0): what proportion of data includes some degree of "misalignment"
        :param variable_hct: boolean: is hct an input parameter to the layer or not?
        """

        # TODO: Are any of these parameters something we might want to infer from data?
        self._gamma = float(system_parameters['gamma'])
        self._b0 = float(system_parameters['b0'])
        self._dchi = float(system_parameters['dchi'])
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

        if not variable_hct:
            self.hct = float(system_parameters['hct'])

        self._full_model = full_model
        self._include_blood = include_blood
        self._misaligned_prob = misaligned_prob
        self._variable_hct = variable_hct

        super().__init__()

    def call(self, input, *args, **kwargs):
        """
        Override the base class call method. This calculate the predicted signal (without added noise)
        given the input OEF/DBV
        :param inputs: a tensor of any shape, where the final dimension has size 2 to represent the OEF/DBV
        :return: The predicted signal
        """
        # Store the original shape, ignoring the last two dimensions
        original_shape = tf.shape(input)
        if self._variable_hct:
            assert input.shape[-1] == 3, 'Input should have 3 elements in last dimension, OEF, DBV and hct'

            # Flatten the inputs except the last axis
            reshaped_input = tf.reshape(input, (-1, 3))
            # Assume oef, dbv and hct are the only elements of the last axis
            oef, dbv, hct = tf.split(reshaped_input, 3, axis=-1)

        else:
            assert input.shape[-1] == 2, 'Input should have 2 elements in last dimension, OEF and DBV'
            # Flatten the inputs except the last axis
            reshaped_input = tf.reshape(input, (-1, 2))
            # Assume oef, dbv and hct are the only elements of the last axis
            oef, dbv = tf.split(reshaped_input, 2, axis=-1)
            hct = self.hct

        if self._misaligned_prob > 0.0:
            # Choose the misaligned examples
            misaligned = tf.random.uniform(oef.shape) < self._misaligned_prob
            # Randomly generate the image index from which slices are misaligned
            misaligned_from_index = tf.random.uniform(oef.shape, minval=4, maxval=(self._taus.shape[0] - 1),
                                                      dtype=tf.dtypes.int32)
            image_indices = tf.range(0, self._taus.shape[0])
            # Create the mask of misaligned images
            misaligned_images_mask = tf.math.logical_and((image_indices > misaligned_from_index), misaligned)
            misaligned_images_mask = tf.cast(misaligned_images_mask, tf.float32)

            # Generate the misaligned oef/dbv
            misaligned_oef = tf.clip_by_value(tf.random.normal(tf.shape(oef), stddev=0.15) + oef, 0.05, 0.8)
            misaligned_dbv = tf.clip_by_value(tf.random.normal(tf.shape(dbv), stddev=0.05) + dbv, 0.002, 0.3)

            oef = oef * (1.0 - misaligned_images_mask) + misaligned_oef * misaligned_images_mask
            dbv = dbv * (1.0 - misaligned_images_mask) + misaligned_dbv * misaligned_images_mask

        tissue_signal = self.calc_tissue(oef, dbv, hct)
        blood_signal = tf.zeros_like(tissue_signal)
        if self._include_blood:
            # Spin densities
            nb = 0.775
            # compartment steady-state magnetization adapted from Cherukara et al code
            # What paper does this come from?
            m_bld = 1 - (2 - tf.math.exp(- (self._tr - self._ti) / self._t1b)) * tf.math.exp(-self._ti / self._t1b)

            blood_weight = m_bld * nb * dbv
            blood_signal = self.calc_blood(oef, hct)
        else:
            blood_weight = dbv

        tissue_weight = 1 - blood_weight

        signal = tissue_weight * tissue_signal + blood_weight * blood_signal

        if self._simulate_noise:
            if signal.shape[-1] == 11:
                # Normalised SNRs are given from real data, and calculated with respect to the tau=0 image
                norm_snr = np.array([0.985, 1.00, 1.01, 1., 0.97, 0.95, 0.93, 0.90, 0.86, 0.83, 0.79], dtype=np.float32)
            elif signal.shape[-1] == 24:
                norm_snr = 1.0 - (np.abs(np.arange(-0.028, 0.065, 0.004)) * 3.0)

            # The actual SNR varies between 60-120, but I've increased the range for more diversity
            snr = tf.random.uniform((signal.shape[0], 1), 50, 120) * tf.reshape(norm_snr, (1, signal.shape[-1]))
            # Calculate the mean signal for each tau value and divie by the snr to get the std-dev
            std_dev = tf.reduce_mean(signal, 0, keepdims=True) / snr
            # Add noise at the correct level
            signal = signal + tf.random.normal(signal.shape) * std_dev

        """
        # Normalise the data based on where tau = 0 to remove arbitrary scaling and take the log
        tau_zero_data = signal[:, tf.where(self._taus == 0)[0][0]]
        signal = tf.math.log(signal/tf.expand_dims(tau_zero_data, 1))
        """

        # The predicted signal should have the original shape with len(self.taus) for the final dimension
        new_shape = tf.concat([original_shape[0:-1], [len(self._taus)]], 0)
        signal = tf.reshape(signal, new_shape)

        return signal

    @staticmethod
    def calculate_dw_static(oef, hct, gamma, b0, dchi):
        return (4.0 / 3.0) * math.pi * gamma * b0 * dchi * hct * oef

    def calculate_dw(self, oef, hct):
        return SignalGenerationLayer.calculate_dw_static(oef, hct, self._gamma, self._b0, self._dchi)

    def calculate_r2p(self, oef, dbv, hct):
        return self.calculate_dw(oef, hct) * dbv

    def calc_tissue(self, oef, dbv, hct):
        """
        :param oef: A tensor containing the oef value of each parameter pair
        :param dbv: A tensor containing the dbv value of each parameter pair
        :return: The signal contribution from brain tissue
        """

        def compose(arg):
            """
            :return: The signal for the given index calculated using the full model
            """
            dbv_i, dw_i = arg
            # lower limit for integration, although it's defined between 0 and 1, 0 gives nans because of divide by 0
            # in integrand....
            a = tf.constant(1e-5, dtype=tf.float32)
            b = tf.constant(1, dtype=tf.float32)  # upper limit for integration
            int_parts = tf.linspace(a, b, 2 ** 7 + 1)
            return tf.math.exp(-dbv_i * integral((2 + int_parts) * tf.math.sqrt(1 - int_parts) * (
                    1.0 - tf.math.special.bessel_j0(1.5 * (tf.expand_dims(self._taus * dw_i, -1)) * int_parts))
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

        dw = self.calculate_dw(oef, hct)

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
            s2 = tf.exp(-self._r2t * self._te) * tf.exp(dbv - (r2p * self._taus))

            signals = s * taus_under_tc + s2 * taus_over_tc

        return signals

    def calc_blood(self, oef, hct):
        """
        :param oef: A tensor containing the the oef values from each parameter pair
        :return: The signal contribution from venous blood
        """
        # Cherukara code
        if False:
            # R2b comes from Golay et al. 2001
            r2b = 4.5 + 16.4 * hct + (165.2 * hct + 55.7) * oef ** 2
            td = 0.0045067
            # Why is the 4pi missing from the squared term here? Missing in cherukara code as well. Also not clear where
            # the 0.14e-6 comes from...
            g0 = (4 / 45) * hct * (1 - hct) * (self._b0 * (self._dchi * oef + 0.14e-6)) ** 2

            signals = tf.math.exp(-r2b * self._te) * tf.math.exp(- (0.5 * (self._gamma ** 2) * g0 * (td ** 2)) *
                                                                 ((self._te / td) + tf.math.sqrt(
                                                                     0.25 + (self._te / td)) + 1.5 -
                                                                  (2.0 * tf.math.sqrt(
                                                                      0.25 + (((self._te + self._taus) ** 2) / td))) -
                                                                  (2.0 * tf.math.sqrt(
                                                                      0.25 + (((self._te - self._taus) ** 2) / td)))))
        # Cherukara maths
        if True:
            # Constants taken from Berman et al. 2018
            r2b = 1.0 / 0.189
            td = (2.6 ** 2.0) / 2.0
            # Convert to seconds
            td = td * 1e-3
            g0 = (4 / 45) * hct * (1 - hct) * (4.0 * math.pi * self._b0 * self._dchi * oef) ** 2

            signals = tf.math.exp(-r2b * self._te) * tf.math.exp(- (0.5 * (self._gamma ** 2) * g0 * (td ** 2)) *
                                                                 ((self._te / td) + tf.math.sqrt(
                                                                     0.25 + (self._te / td)) + 1.5 -
                                                                  (2.0 * tf.math.sqrt(
                                                                      0.25 + ((self._te + self._taus) / td))) -
                                                                  (2.0 * tf.math.sqrt(
                                                                      0.25 + ((self._te - self._taus) / td)))))
        return signals


def create_synthetic_dataset(params, full_model, use_blood, misaligned_prob, variable_hct=False, uniform_prop=0.1):
    import tensorflow_probability as tfp
    sig_layer = SignalGenerationLayer(params, full_model, use_blood, misaligned_prob=misaligned_prob,
                                      variable_hct=variable_hct)
    oefs = tf.random.uniform((round(int(params['sample_size']) * uniform_prop),), minval=float(params['oef_start']),
                             maxval=float(params['oef_end']))

    oefs_n = tf.random.normal((round(int(params['sample_size']) * (1.0 - uniform_prop)),)) * float(
        params['oef_std']) + float(params['oef_mean'])
    oefs_n = tf.clip_by_value(oefs_n, float(params['oef_start']), float(params['oef_end']))
    oefs = tf.concat([oefs, oefs_n], 0)

    dbvs = tf.random.uniform((round(int(params['sample_size']) * uniform_prop),), minval=float(params['dbv_start']),
                             maxval=float(params['dbv_end']))

    dbvs_n = tf.cast(tfp.distributions.TruncatedNormal(loc=float(params['dbv_mean']), scale=float(params['dbv_std']),
                                                       low=float(params['dbv_start']), high=float(params['dbv_end'])).
                     sample((round(int(params['sample_size']) * (1.0 - uniform_prop)),)), tf.float32)
    dbvs = tf.concat([dbvs, dbvs_n], 0)

    xx, yy = tf.meshgrid(oefs, dbvs, indexing='ij')
    train_y = tf.stack([tf.reshape(xx, [-1]), tf.reshape(yy, [-1])], axis=1)

    if variable_hct:
        hcts = tf.random.uniform((train_y.shape[0], 1), minval=0.34, maxval=0.34)

        train_y = tf.concat([train_y, hcts], axis=-1)

    # Remove any ordering from the data
    train_y = tf.random.shuffle(train_y)
    train_x_list = []
    # break into chunks to avoid running out of memory
    for i in range(10):
        chunk_size = train_y.shape[0] // 10
        y_subset = train_y[i * chunk_size:(i + 1) * chunk_size]
        train_x_list.append(sig_layer(y_subset))

    train_x = tf.concat(train_x_list, axis=0)

    # Convert Hct and OEF into dHb (deoxyhameoglobin concentration) for extra training signals
    # dhbs = train_y[:, 0] * (train_y[:, 2] / 0.03)

    # Calculate R2' for extra training signals
    if variable_hct:
        r2p = sig_layer.calculate_r2p(train_y[:, 0], train_y[:, 1], train_y[:, 2])
    else:
        r2p = sig_layer.calculate_r2p(train_y[:, 0], train_y[:, 1], sig_layer.hct)

    # Concatenate the R2'
    train_y = tf.concat([train_y[:, :2], tf.expand_dims(r2p, -1)], -1)
    return train_x, train_y


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

    _train_x, _train_y = create_synthetic_dataset(params, args.f, args.b, 0.1, False)

    np.savez('synthetic_data', x=_train_x, y=_train_y)
