# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Do analysis on the data post training
import os
import nibabel as nib
import scipy.stats as ss
import numpy as np

file_types = ['oef', 'dbv', 'r2p']
conditions = ['baseline', 'hyperv']

def calculate_smooth_name(data_dir,  file_base):
    return data_dir + '/' + file_base + '_smooth.nii.gz'

def smooth_data(data_dir, gauss_size=4.0):
    for condition in conditions:
        for file_type in file_types:
            file_base = condition + '_' + file_type
            cmd = 'fslmaths '+data_dir+'/' + file_base + '.nii.gz -kernel gauss '+ str(gauss_size) + ' -fmean ' +\
                calculate_smooth_name(data_dir, file_base)
            os.system(cmd)


def calculate_t_test(data_dir):
    for file_type in file_types:
        file1 = calculate_smooth_name(data_dir, conditions[0] + '_' + file_type)
        file2 = calculate_smooth_name(data_dir, conditions[1] + '_' + file_type)

        nib1 = nib.load(file1)
        nib2 = nib.load(file2)
        data1 = np.reshape(nib1.get_fdata(), (-1, 6))
        data2 = np.reshape(nib2.get_fdata(), (-1, 6))

        indices = np.sum(np.abs(data1 - data2), -1) > 1e-5
        indices = indices * (np.min(data1, -1) > 5e-2) * (np.min(data2, -1) > 5e-2)
        data1 = data1[indices, :]
        data2 = data2[indices, :]

        t_data, p_data = ss.ttest_rel(data1, data2, axis=1,
                     nan_policy='omit')

        t_results = np.zeros((indices.shape[0]))
        p_results = np.zeros((indices.shape[0]))

        t_results[indices] = t_data
        p_results[indices] = p_data

        t_results = np.reshape(t_results, nib1.get_fdata().shape[0:3] + (1,))

        new_header = nib1.header.copy()
        array_img = nib.Nifti1Image(t_results, affine=None, header=new_header)
        nib.save(array_img, data_dir + '/' + file_type + '_t_test.nii.gz')


if __name__ == '__main__':
    data_dir = '/Users/is321/Documents/Data/qBold/model_predictions/optimaltruncnormalmulti/'
    smooth_data(data_dir, 6)

    calculate_t_test(data_dir)
