#!/usr/bin/env python3

from tqdm import tqdm

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

    ln_Sase = np.log(signals)
    ln_Sase[np.where(np.isnan(ln_Sase))] = 0
    ln_Sase[np.where(np.isinf(ln_Sase))] = 0

    p = np.zeros((*signals.shape[:-1], 2))

    for vID in range(signals.shape[0]):  # takes roughly 1 min per volume

        tau_lineID = np.where(taus > 0.016)
        w = 1 / taus[tau_lineID]

        for xID in tqdm(range(p.shape[1])):
            for yID in range(p.shape[2]):
                for zID in range(p.shape[3]):
                    X = taus[tau_lineID]
                    X = np.vstack((X, np.ones_like(X))).T
                    Y = np.squeeze(ln_Sase[vID, xID, yID, zID, tau_lineID])

                    wls = LinearRegression()
                    wls.fit(X, Y, sample_weight=w)

                    p[vID, xID, yID, zID] = [wls.coef_[0], wls.intercept_]

    s0_id = np.where(taus == 0)[0]

    r2p = -p[:, :, :, :, 0:1]
    c = p[:, :, :, :, 1:]

    dbv = c - ln_Sase[:, :, :, :, s0_id]
    oef = r2p / (dbv * gamma * (4 / 3) * np.pi * dchi * hct * b0)
    dhb = r2p / (dbv * gamma * (4 / 3) * np.pi * dchi * b0 * 0.03)

    save_predictions(p, 'baseline')
