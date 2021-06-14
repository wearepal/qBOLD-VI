#!/usr/bin/env python3

import numpy as np
import argparse
import configparser
from tqdm import tqdm

import scipy.integrate as integrate
import scipy.special as special
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
        self._taus = np.arange(float(system_parameters['tau_start']), float(system_parameters['tau_end']),
                               float(system_parameters['tau_step']), dtype=np.float32)

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
        # TODO: implement the other parts of the signal model

        signal = tissue_signal
        # The predicted signal should have the original shape with len(self.taus)
        signal = tf.reshape(signal, original_shape+(len(self._taus,)))
        return signal

    def calc_tissue(self, oef, dbv):
        dw = (4 / 3) * np.pi * self._gamma * self._b0 * self._dchi * self._hct * oef
        tc = 1 / dw
        r2p = dw * dbv

        if self._full_model:
            # TODO: Implement the full model
            # Tensorflow has some solvers https://www.tensorflow.org/probability/api_docs/python/tfp/math/ode
            # some experimentation might be required
            """
            s = integrate.quad(lambda u: (2 + u) * np.sqrt(1 - u) *
                                             (1 - special.jv(0, 1.5 * (tau * dw) * u)) / (u ** 2), 0, 1)[0]

                s = np.exp(-dbv * s / 3)
                s *= np.exp(-self._te * self._r2t)
                signals[i] = s
            """
            raise NotImplementedError()
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


def calc_tissue(params, full, oef, dbv):
    taus = np.arange(float(params['tau_start']), float(params['tau_end']), float(params['tau_step']))
    gamma = float(params['gamma'])
    b0 = float(params['b0'])
    dchi = float(params['dchi'])
    hct = float(params['hct'])
    te = float(params['te'])
    r2t = float(params['r2t'])

    dw = (4 / 3) * np.pi * gamma * b0 * dchi * hct * oef
    tc = 1 / dw
    r2p = dw * dbv

    signals = np.zeros_like(taus)

    for i, tau in enumerate(taus):
        if full:
            s = integrate.quad(lambda u: (2 + u) * np.sqrt(1 - u) *
                                         (1 - special.jv(0, 1.5 * (tau * dw) * u)) / (u ** 2), 0, 1)[0]

            s = np.exp(-dbv * s / 3)
            s *= np.exp(-te * r2t)
            signals[i] = s
        else:
            if abs(tau) < tc:
                s = np.exp(-r2t * te) * np.exp(- (0.3 * (r2p * tau) ** 2) / dbv)
            else:
                s = np.exp(-r2t * te) * np.exp(dbv - (r2p * abs(tau)))
            signals[i] = s

    return signals


def calc_blood(params, oef):
    hct = float(params['hct'])
    dchi = float(params['dchi'])
    b0 = float(params['b0'])
    te = float(params['te'])
    gamma = float(params['gamma'])

    r2b = 4.5 + 16.4 * hct + (165.2 * hct + 55.7) * oef ** 2

    td = 0.0045067
    g0 = (4 / 45) * hct * (1 - hct) * ((dchi * b0) ** 2)

    signals = np.zeros_like(taus)

    for i, tau in enumerate(taus):
        signals[i] = np.exp(-r2b * te) * np.exp(- (0.5 * (gamma ** 2) * g0 * (td ** 2)) *
                                                ((te / td) + np.sqrt(0.25 + (te / td)) + 1.5 -
                                                 (2 * np.sqrt(0.25 + (((te + taus[i]) ** 2) / td))) -
                                                 (2 * np.sqrt(0.25 + (((te - taus[i]) ** 2) / td)))))

    return signals


def generate_signal(params, full, include_blood, oef, dbv):
    tissue_signal = calc_tissue(params, full, oef, dbv)
    blood_signal = 0

    tr = float(params['tr'])
    ti = float(params['ti'])
    t1b = float(params['t1b'])
    s0 = int(params['s0'])

    if include_blood:
        nb = 0.775

        m_bld = 1 - (2 - np.exp(- (tr - ti) / t1b)) * np.exp(-ti / t1b)

        vb = dbv

        blood_weight = m_bld * nb * vb
        tissue_weight = 1 - blood_weight

        blood_signal = calc_blood(params, oef)
    else:
        blood_weight = dbv
        tissue_weight = 1 - blood_weight

    return s0 * (tissue_weight * tissue_signal + blood_weight * blood_signal)


def test_layer_matches_python():
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']

    sig_layer = SignalGenerationLayer(params, False, False)

    test_oef = tf.random.uniform((2,), minval=float(params['oef_start']), maxval=float(params['oef_end']))
    test_dbv = tf.random.uniform((2,), minval=float(params['dbv_start']), maxval=float(params['dbv_end']))

    test_oef_dbv = tf.stack([test_oef, test_dbv], axis=-1)

    # TODO: Make the test cover all the variants (blood/ fully model)
    # Current the model only considers the tissue contribution
    signal = sig_layer(test_oef_dbv)
    signal2 = calc_tissue(params, False, test_oef[0], test_dbv[0])

    assert (tf.keras.backend.mean(signal[0]-signal2) < 1e-4), "Predictions are above epislon different"



if __name__ == '__main__':
    test_layer_matches_python()


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

    taus = np.arange(float(params['tau_start']), float(params['tau_end']), float(params['tau_step']))

    snr = float(params['snr'])

    oefs = np.linspace(float(params['oef_start']), float(params['oef_end']), int(params['sample_size']))
    dbvs = np.linspace(float(params['dbv_start']), float(params['dbv_end']), int(params['sample_size']))
    train_y = np.array(np.meshgrid(oefs, dbvs)).T.reshape(-1, 2)
    train_x = np.zeros((oefs.size * dbvs.size, taus.size))
    for i, [oef, dbv] in enumerate(tqdm(train_y)):
        train_x[i] = generate_signal(params, args.f, args.b, oef, dbv)

        se = train_x[i][np.where(taus == 0)]
        train_x[i] /= se
        train_x[i] = np.log(train_x[i])

        stdd = abs(train_x[i].mean()) / snr
        train_x[i] += np.random.normal(0, stdd, train_x[i].size)

    train = np.hstack((train_x, train_y))
    np.random.shuffle(train)
    train_x = train[:, :11]
    train_y = train[:, 11:]

    np.savetxt('signals.csv', train_x, delimiter=',')
    np.savetxt('params.csv', train_y, delimiter=',')
