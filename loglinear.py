#!/usr/bin/env python3

from tqdm import tqdm

import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
import configparser

tau = np.arange(-16, 65, 8) * 10 ** -3
Hct = 0.34
dChi0 = 0.264 * 10 ** -6
gamma = 2.675 * 10 ** 4
B0 = 3 * 10 ** 4
phi = 1.34
k = 0.03


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


def ase_qbold_3d(ln_Sase):
    ln_Sase[np.where(np.isnan(ln_Sase))] = 0
    ln_Sase[np.where(np.isinf(ln_Sase))] = 0

    Tc = 0.016

    tau_lineID = np.where(tau > Tc)
    w = 1 / tau[tau_lineID]
    p = np.zeros((*ln_Sase.shape[:-1], 2))

    for xID in tqdm(range(p.shape[0])):
        for yID in range(p.shape[1]):
            for zID in range(p.shape[2]):
                X = tau[np.where(tau > 0.017)]
                X = np.vstack((X, np.ones_like(X))).T
                Y = np.squeeze(ln_Sase[xID, yID, zID, tau_lineID])

                wls = LinearRegression()
                wls.fit(X, Y, sample_weight=w)

                p[xID, yID, zID] = [wls.coef_[0], wls.intercept_]

    return p


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']
    parser = argparse.ArgumentParser(description='Estimate parameters using log-linear method')

    parser.add_argument('-f', default='baseline_ase.npy', help='path to signal data')

    args = parser.parse_args()

    signals = np.load(args.f)[:1]
    ln_Sase = np.log(signals)

    p = np.zeros((*signals.shape[:-1], 2))

    for i in range(signals.shape[0]):
        p[i] = ase_qbold_3d(ln_Sase[i])

    s0_id = np.where(tau == 0)[0]

    r2p = -p[:, :, :, :, 0:1]
    c = p[:, :, :, :, 1:]

    dbv = c - ln_Sase[:, :, :, :, s0_id]
    oef = r2p / (dbv * gamma * (4/3) * np.pi * dChi0 * Hct * B0)
    dhb = r2p / (dbv * gamma * (4/3) * np.pi * dChi0 * B0 * k)

    save_predictions(p, 'baseline')
