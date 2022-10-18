#!/usr/bin/env python3

import argparse
import configparser
import os

import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def save_predictions(predictions, filename, transform_directory):
    def save_im_data(im_data, _filename):
        images = np.split(im_data, im_data.shape[0], axis=0)
        images = np.squeeze(np.concatenate(images, axis=-1), 0)

        if transform_directory is not None:
            existing_nib = nib.load(transform_directory + '/example.nii.gz')
            new_header = existing_nib.header.copy()
            array_img = nib.Nifti1Image(images, None, header=new_header)
        else:
            array_img = nib.Nifti1Image(images, None)

        nib.save(array_img, _filename + '.nii.gz')

    oef = predictions[0]
    dbv = predictions[1]
    r2p = predictions[2]

    if transform_directory:
        import os
        mni_ims = filename + '_merged.nii.gz'
        merge_cmd = 'fslmerge -t ' + mni_ims
        ref_image = transform_directory + '/MNI152_T1_2mm.nii.gz'
        for i in range(oef.shape[0]):
            nonlin_transform = transform_directory + '/nonlin' + str(i) + '.nii.gz'
            oef_im = oef[i, ...]
            dbv_im = dbv[i, ...]
            r2p_im = r2p[i, ...]
            subj_ims = np.stack([oef_im, dbv_im, r2p_im], 0)

            subj_im = filename + '_subj' + str(i)
            save_im_data(subj_ims, subj_im)
            subj_im_mni = subj_im + 'mni'
            # Transform
            cmd = 'applywarp --in=' + subj_im + ' --out=' + subj_im_mni + ' --warp=' + nonlin_transform + \
                  ' --ref=' + ref_image
            os.system(cmd)
            merge_cmd = merge_cmd + ' ' + subj_im_mni

        os.system(merge_cmd)
        merged_nib = nib.load(mni_ims)
        merged_data = merged_nib.get_fdata()

        file_types = ['_oef_mni', '_dbv_mni', '_r2p_mni']
        for t_idx, t in enumerate(file_types):
            t_data = merged_data[:, :, :, t_idx::3]
            new_header = merged_nib.header.copy()
            array_img = nib.Nifti1Image(t_data, affine=None, header=new_header)
            nib.save(array_img, filename + t + '.nii.gz')

    save_im_data(oef, filename + '_oef')
    save_im_data(dbv, filename + '_dbv')
    save_im_data(r2p, filename + '_r2p')


def fit_wls(signals):
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

    oef = np.clip(oef, 0.01, 0.8)
    dbv = np.clip(dbv, 0.002, 0.25)
    r2p = np.clip(r2p, 1e-2, 100)
    return oef, dbv, r2p

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
    taus = np.around(np.arange(float(params['tau_start']), float(params['tau_end']),
                               float(params['tau_step']), dtype=np.float32), decimals=7)

    data_dir = '/home/data/qbold/'
    op_dir = 'wls_clip'
    if os.path.exists(op_dir) == False:
        os.mkdir(op_dir)

    """
    hyperv_data = np.load(data_dir+'/hyperv_ase.npy')
    hyperv_data = hyperv_data[:, :, :, :, :-2]
    baseline_data = np.load(data_dir+'/baseline_ase.npy')
    baseline_data = baseline_data[:, :, :, :, :-2]

    transform_dir_baseline = data_dir + '/transforms_baseline/'
    transform_dir_hyperv = data_dir+ '/transforms_hyperv/'

    oef, dbv, r2p = fit_wls(baseline_data)
    save_predictions([oef, dbv, r2p], op_dir+'/baseline', transform_directory=transform_dir_baseline)
    oef, dbv, r2p = fit_wls(hyperv_data)
    save_predictions([oef, dbv, r2p], op_dir+'/hyperv', transform_directory=transform_dir_hyperv)
    """

    params['tau_start'] = '-0.028'
    params['tau_step'] = '0.004'
    taus = np.around(np.arange(float(params['tau_start']), float(params['tau_end']),
                     float(params['tau_step']), dtype=np.float32), decimals=7)
    streamlined_data = np.load(data_dir + '/streamlined_ase.npy')

    oef, dbv, r2p = fit_wls(streamlined_data[:, :, :, :, :-2])

    save_predictions([oef, dbv, r2p], op_dir + '/streamlined', transform_directory=None)

