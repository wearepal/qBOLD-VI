# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Create the figures for the paper

import sys, os, argparse, yaml
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from train import setup_argparser, get_defaults

root_exp_dir = '/Users/is321/Documents/Data/qBold/model_predictions/'


baseline_gm_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/baseline_ase_gm.nii.gz'
hyperv_gm_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/hyperv_ase_gm.nii.gz'
mni_image = '/Users/is321/Documents/Data/qBold/hyperv_data/MNI152_t1_2mm.nii.gz'

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

def get_tstat_x(x, exp_directory, pt=False):
    assert pt==False
    filename = exp_directory + '/' + x + '_mni_t_test.nii.gz'
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


def compare_conditions_all(exp_tuple, save_name=None):
    xs = ['oef', 'dbv', 'r2p']
    fig, axis = plt.subplots(nrows=1, ncols=len(xs), figsize=(9, 4))
    for idx, x in enumerate(xs):
        ax = axis[idx]
        data = [get_masked_x(x, root_exp_dir + '/' + exp_tuple[0], True)]
        data.append(get_masked_x(x, root_exp_dir + '/' + exp_tuple[0], False))
        ax.violinplot(data, showmedians=True, showextrema=False)
        ax.set_ylabel(x)
        set_violin_axis_style(ax, [('','baseline'), ('','hyperventilation')])
        ax.set_ylim(np.min(data[0]), np.percentile(data[0], 98))
    plt.suptitle(exp_tuple[1])
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name)
    else:
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
    plt.suptitle(x)
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def show_t_statistcs(x, save_name=None):
    from post_analysis import calculate_t_test, smooth_data
    subj_indicies = [0, 3, 4]
    slice_indicies = [45, 40, 35, 30]
    fig = plt.figure(figsize=(len(experiment_names)*2 + 1, len(slice_indicies)*1.6))
    axis = fig.subplots(nrows=len(slice_indicies), ncols=len(experiment_names))
    vmin = 2.5
    vmax = 5
    mni = nib.load(mni_image)
    mni_data = mni.get_fdata()

    for idx, exp_name in enumerate(experiment_names):
        exp_dir = root_exp_dir + '/' + exp_name[0]
        #smooth_data(exp_dir, 6)
        #calculate_t_test(exp_dir)
        data = get_tstat_x(x, exp_dir)
        for slice_idx in range(len(slice_indicies)):
            ax = axis[slice_idx, idx]
            slice = data[15:-15, 10:-10, slice_indicies[slice_idx], 0]
            mni_slice = mni_data[15:-15, 10:-10, slice_indicies[slice_idx]]

            ax.imshow(mni_slice, cmap='gray')

            slice_pos_mask = (slice > vmin) * 1.0
            slice_pos = slice * slice_pos_mask
            ax.imshow(slice_pos, alpha=slice_pos_mask, vmin=vmin, vmax=vmax, cmap='hot')
            im = ax.imshow(slice_pos, alpha=0.0, vmin=vmin, vmax=vmax, cmap='hot')

            slice_neg_mask = (slice < -vmin) * 1.0
            slice_neg = slice * slice_neg_mask
            ax.imshow(slice_neg, alpha=slice_neg_mask, vmin=-vmax, vmax=-vmin, cmap='cool')
            im2 = ax.imshow(slice_neg, alpha=0.0, vmin=-vmax, vmax=-vmin, cmap='cool')
            ax.axis('off')
            if slice_idx == 0:
                ax.set_title(exp_name[1])
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.9, 0.55, 0.05, 0.35])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_alpha(1)
        cbar.draw_all()

        cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.35])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_alpha(1)
        cbar.draw_all()
        plt.suptitle(x + ' t-statistic')
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

    show_t_statistcs('oef', save_name='likelihood_tstat_oef.pdf')
    show_t_statistcs('dbv', save_name='likelihood_tstat_dbv.pdf')
    show_t_statistcs('r2p', save_name='likelihood_tstat_r2p.pdf')

# Forward model experiments
if False:
    experiment_names = [('nobloodnofull_tv1', 'NB/A'), ('nofull_tv1', 'B/A'), ('noblood_tv1', 'NB/F'), ('data_norm_multi', 'B/F')]

    compare_experiments_baseline_all(save_name='forwardmodels_stats.pdf')
    # show_example_slices_x('dbv', True, save_name='multi_single_examples.pdf')
    compare_experiments_masked_elbo(save_name='forwardmodels_elbo.pdf')

    experiment_names = [('noblood_tv1', 'NB/F'),
                        ('data_norm_multi', 'B/F')]
    compare_conditions_all(experiment_names[0], 'nbf_condition_dist.pdf')
    compare_conditions_all(experiment_names[1], 'bf_condition_dist.pdf')

    show_t_statistcs('oef', save_name='forwardmodels_tstat_oef.pdf')
    show_t_statistcs('dbv', save_name='forwardmodels_tstat_dbv.pdf')
    show_t_statistcs('r2p', save_name='forwardmodels_tstat_r2p.pdf')
#compare_x_conditions('r2p', root_exp_dir + '/' + experiment_names[0])
# Smoothness
if False:
    experiment_names = [('data_norm_multi_tv0', 'TV=0'),
                        ('data_norm_multi', 'TV=1'),
                        ('data_norm_multi_tv2', 'TV=2'),
                        ('data_norm_multi_tv5', 'TV=5')]
    show_t_statistcs('oef', save_name='smoothness_tstat_oef.pdf')
    show_t_statistcs('dbv', save_name='smoothness_tstat_dbv.pdf')
    show_t_statistcs('r2p', save_name='smoothness_tstat_r2p.pdf')

    show_example_slices_x('dbv', True, save_name='smoothness_examples_dbv.pdf')
    show_example_slices_x('oef', True, save_name='smoothness_examples_oef.pdf')
    show_example_slices_x('r2p', True, save_name='smoothness_examples_r2p.pdf')

if True:
    def create_pt_exp(old_dir, new_dir):
        from glob import  glob
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        pt_files = glob(old_dir+'/pt_*')
        for pt_file in pt_files:
            bname = os.path.basename(pt_file)
            bname = bname[3:]
            target_filename = new_dir + '/' + bname
            if not os.path.exists(target_filename):
                cmd = 'cp '+pt_file + ' ' + target_filename
                os.system(cmd)
    create_pt_exp(root_exp_dir+'/'+'optimal2', root_exp_dir+'/'+'optimal2_pt')
    experiment_names = [('wls_clip', 'WLS'),
                        ('optimal2_pt', 'Synth'),
                        ('optimal2_tv0', 'VI'),
                        ('optimal2', 'VI + TV')]
    show_t_statistcs('oef', save_name='vi_synth_tstat_oef.pdf')
    show_t_statistcs('dbv', save_name='vi_synth_tstat_dbv.pdf')
    show_t_statistcs('r2p', save_name='vi_synth_tstat_r2p.pdf')

    show_example_slices_x('dbv', True, save_name='vi_syth_wls_examples_dbv.pdf')
    show_example_slices_x('oef', True, save_name='vi_syth_wls_examples_oef.pdf')
    show_example_slices_x('r2p', True, save_name='vi_syth_wls_examples_r2p.pdf')