#!/usr/bin/env python3

import numpy as np
import argparse
import configparser
from tqdm import tqdm

import scipy.integrate as integrate
import scipy.special as special


def calc_tissue(params, full, oef, dbv):
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
