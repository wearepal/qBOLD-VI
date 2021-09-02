# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Create the figures for the paper

import sys, os, argparse, yaml
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from train import setup_argparser, get_defaults

root_exp_dir = '/Users/is321/Documents/Data/qBold/model_predictions/'
experiment_names = ['data_norm_multi_tv0','data_norm_multi', 'data_norm_multi_tv2', 'data_norm_multi_tv3', 'data_norm_multi_tv5',
                    'data_norm_multi_tv10']

baseline_gm_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/baseline_ase_gm.nii.gz'
hyperv_gm_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/hyperv_ase_gm.nii.gz'

def get_mask_data(baseline):
    if baseline:
        mask = nib.load(baseline_gm_masks)
    else:
        mask = nib.load(hyperv_gm_masks)
    return mask.get_fdata()

def get_data_x(x, exp_directory, baseline=True, pt=False):
    filename = exp_directory + '/'
    if pt:
        filename = filename + 'pt_'
    if baseline:
        filename = filename + 'baseline_'
    else:
        filename = filename + 'hyperv_'

    filename = filename + x + '.nii.gz'

    img = nib.load(filename)
    data = img.get_fdata()
    return data

def get_masked_x(x, exp_directory, baseline=True, pt=False):
    data = get_data_x(x, exp_directory, baseline, pt)

    mask = get_mask_data(baseline)

    return data.flatten()[mask.flatten() == 1.0]


def get_masked_elbo(exp_directory, baseline=True):
    likelihood = get_masked_x('likelihood', exp_directory, baseline=baseline)
    kl = get_masked_x('kl', exp_directory, baseline=baseline)
    return likelihood+kl

def get_masked_elbo_both(exp_directory):
    return np.concatenate([get_masked_elbo(exp_directory, True), get_masked_elbo(exp_directory, False)])

def set_violin_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    labels = [x[1] for x in labels]
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def compare_x_conditions(x, exp_directory):
    baseline_x = get_masked_x(x, exp_directory, True)
    hyperv_x = get_masked_x(x, exp_directory, False)

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax1.violinplot([baseline_x, hyperv_x])
    set_violin_axis_style(ax1, ['baseline', 'hyperventilation'])
    ax1.set_ylabel(x)
    plt.tight_layout()
    plt.show()


def compare_experiments_baseline_all(save_name=None):
    xs = ['oef', 'dbv', 'r2p']
    fig, axis = plt.subplots(nrows=1, ncols=len(xs), figsize=(9, 4))
    for idx, x in enumerate(xs):
        ax = axis[idx]
        data = []
        for exp in experiment_names:
            data.append(get_masked_x(x, root_exp_dir + '/' + exp[0], True))
        ax.violinplot(data, showmedians=True, showextrema=False)
        ax.set_ylabel(x)
        set_violin_axis_style(ax, experiment_names)
        ax.set_ylim(np.min(data), np.percentile(data, 98))
    plt.tight_layout()
    """
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    quartile1, medians, quartile3 = np.percentile(elbos, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(elbos, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax1.plot(inds, medians, marker='o', color='k', ms=5, zorder=3)
    ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax1.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)"""
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()

def compare_experiments_masked_elbo(save_name=None):
    elbos = []
    for exp in experiment_names:
        elbos.append(get_masked_elbo_both(root_exp_dir + '/' + exp[0]))

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(len(experiment_names)*2+1, 4))
    ax1.violinplot(elbos, showmedians=True, showextrema=False)
    """
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
    quartile1, medians, quartile3 = np.percentile(elbos, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(elbos, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax1.plot(inds, medians, marker='o', color='k', ms=5, zorder=3)
    ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax1.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)"""
    set_violin_axis_style(ax1, experiment_names)
    ax1.set_ylim(np.min(elbos), np.percentile(elbos, 98))
    ax1.set_ylabel('GM Voxelwise ELBO')
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def show_example_slices_x(x, baseline, save_name=None):
    subj_indicies = [0, 3, 4]
    slice_indicies = [4, 5, 4]
    fig = plt.figure(figsize=(len(experiment_names)*2 + 1, len(subj_indicies)*2))
    axis = fig.subplots(nrows=len(subj_indicies), ncols=len(experiment_names))
    vmin = None
    vmax = None
    if x == 'oef':
        vmin = 0.05
        vmax = 0.5
    if x == 'dbv':
        vmin = 0.0
        vmax = 0.2
    if x == 'r2p':
        vmin = 0.0
        vmax = 10

    for idx, exp_name in enumerate(experiment_names):
        exp_dir = root_exp_dir + '/' + exp_name[0]
        data = get_data_x(x, exp_dir, baseline)
        for slice_idx in range(len(subj_indicies)):
            ax = axis[slice_idx, idx]
            im = ax.imshow(data[15:-15, 10:-10, slice_indicies[slice_idx], subj_indicies[slice_idx]], vmin=vmin, vmax=vmax)
            ax.axis('off')
            ax.set_title(exp_name[1])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()

# Data normalisation experiments
if False:
    experiment_names = [('data_norm_single', 'single'), ('data_norm_multi', 'multi')]

    compare_experiments_baseline_all(save_name='multi_single_stats.pdf')
    #show_example_slices_x('dbv', True, save_name='multi_single_examples.pdf')
    compare_experiments_masked_elbo(save_name='multi_single_elbo.pdf')

# Likelihood experiments
if False:
    experiment_names = [('gauss_tv1', 'Gaussian'), ('data_norm_multi', 'Student-T (2)')]

    compare_experiments_baseline_all(save_name='likelihood_stats.pdf')
    # show_example_slices_x('dbv', True, save_name='multi_single_examples.pdf')
    compare_experiments_masked_elbo(save_name='likelihood_elbo.pdf')

# Likelihood experiments
if False:
    experiment_names = [('nobloodnofull_tv1', 'NB/A'), ('nofull_tv1', 'B/A'), ('noblood_tv1', 'NB/F'), ('data_norm_multi', 'B/F')]

    compare_experiments_baseline_all(save_name='forwardmodels_stats.pdf')
    # show_example_slices_x('dbv', True, save_name='multi_single_examples.pdf')
    compare_experiments_masked_elbo(save_name='forwardmodels_elbo.pdf')
#compare_x_conditions('r2p', root_exp_dir + '/' + experiment_names[0])

