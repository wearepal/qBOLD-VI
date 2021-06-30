#!/usr/bin/env python3

import nibabel as nib
import tensorflow as tf

import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
import configparser


def save_predictions(predictions, filename):
    import nibabel as nib

    def save_im_data(im_data, _filename):
        images = np.split(im_data, predictions.shape[0], axis=0)
        images = np.squeeze(np.concatenate(images, axis=-1), 0)
        affine = np.eye(4)
        array_img = nib.Nifti1Image(images, affine)
        nib.save(array_img, _filename + '.nii.gz')

    save_im_data(predictions[:, :, :, :, 0:1], filename + '_oef_ll')
    save_im_data(predictions[:, :, :, :, 1:2], filename + '_dbv_ll')


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']
    parser = argparse.ArgumentParser(description='Estimate parameters using log-linear method')

    parser.add_argument('-f', default='baseline_ase.npy', help='path to signal data')

    args = parser.parse_args()

    dchi = float(params['dchi'])
    gamma = float(params['gamma'])
    b0 = float(params['b0'])
    hct = float(params['hct'])
    te = float(params['te'])
    r2t = float(params['r2t'])
    tr = float(params['tr'])
    ti = float(params['ti'])
    t1b = float(params['t1b'])
    taus = np.arange(float(params['tau_start']), float(params['tau_end']),
                     float(params['tau_step']), dtype=np.float32)

    signals = np.load(args.f)

    w = 1 / taus[np.where(taus > 0.016)]
    x = taus[np.where(taus > 0.016)]
    x = np.vstack((x, np.ones_like(x))).T

    def map_fn(signal):
        if signal[-1] == 0:
            return 0, 0

        signal = signal[:-1]
        signal /= signal[2]  # tried both with and without
        signal = np.log(signal)  # tried both with and without

        y = signal[np.where(taus > 0.016)]

        wls = LinearRegression()
        wls.fit(x, y, sample_weight=w)

        m = wls.coef_[0]
        c = wls.intercept_

        r2p = -m
        dbv = abs(c - signal[2])  # try with and without abs
        oef = (3 * r2p) / (4 * np.pi * gamma * b0 * dchi * hct * dbv)

        return np.array([oef, dbv])

    p = np.apply_along_axis(map_fn, 4, signals)  # takes long time for all volumes try signals[0:1]
    p = np.clip(p, a_min=1e-3, a_max=0.99)

    save_predictions(p, 'baseline')
