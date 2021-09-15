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
baseline_tmean = '/Users/is321/Documents/Data/qBold/hyperv_data/baseline_ase_tmean.nii.gz'
hyperv_tmean = '/Users/is321/Documents/Data/qBold/hyperv_data/hyperv_ase_tmean.nii.gz'
mni_image = '/Users/is321/Documents/Data/qBold/hyperv_data/MNI152_t1_2mm.nii.gz'
baseline_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/baseline_ase_mask.nii.gz'
hyperv_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/hyperv_ase_mask.nii.gz'

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

        print(exp_tuple[1], x, np.mean(data[0]), np.std(data[0]), np.mean(data[1]), np.std(data[1]))
    plt.suptitle(exp_tuple[1])
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def compare_experiments_baseline_all(save_name=None, plot_line=False, x_label=None):
    xs = ['oef', 'dbv', 'r2p']
    fig, axis = plt.subplots(nrows=1, ncols=len(xs), figsize=(9, 4))
    for idx, x in enumerate(xs):
        ax = axis[idx]
        data = []
        medians = []
        for exp in experiment_names:
            data.append(get_masked_x(x, root_exp_dir + '/' + exp[0], True))
            medians.append(np.median(data[-1]))
        ax.violinplot(data, showmedians=True, showextrema=False)
        ax.set_ylabel(x)
        if x_label:
            ax.set_xlabel(x_label)

        if plot_line:
            ax.plot(1+np.arange(len(experiment_names)), np.array(medians))
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


def show_example_slices_x(x, baseline, save_name=None, last_two=False):
    subj_indicies = [0, 1, 5, 2]
    slice_indicies = [6, 5, 3, 2]

    if last_two:
        subj_indicies = subj_indicies[-2:]
        slice_indicies = slice_indicies[-2:]

    im_filename = hyperv_tmean
    mask_filename = hyperv_masks
    if baseline:
        im_filename = baseline_tmean
        mask_filename = baseline_masks

    mask_nib = nib.load(mask_filename)
    masks = mask_nib.get_fdata()

    im_nib = nib.load(im_filename)
    tmean_images = im_nib.get_fdata()
    fig = plt.figure(figsize=((len(experiment_names)+1)*2 + 1, len(subj_indicies)*1.6))
    axis = fig.subplots(nrows=len(subj_indicies), ncols=len(experiment_names)+1)
    vmin = None
    vmax = None
    if x == 'oef':
        vmin = 0.1
        vmax = 0.5
    if x == 'dbv':
        vmin = 0.0
        vmax = 0.1
    if x == 'r2p':
        vmin = 0.0
        vmax = 10

    for slice_idx in range(len(subj_indicies)):
        ax = axis[slice_idx, 0]
        im = ax.imshow(tmean_images[15:-15, 10:-10, slice_indicies[slice_idx], subj_indicies[slice_idx]], cmap='gray')
        ax.axis('off')

    for idx, exp_name in enumerate(experiment_names):
        exp_dir = root_exp_dir + '/' + exp_name[0]
        data = get_data_x(x, exp_dir, baseline)
        data = data * masks
        for slice_idx in range(len(subj_indicies)):
            ax = axis[slice_idx, idx+1]
            im = ax.imshow(data[15:-15, 10:-10, slice_indicies[slice_idx], subj_indicies[slice_idx]], vmin=vmin, vmax=vmax)
            ax.axis('off')
            if slice_idx == 0:
                ax.set_title(exp_name[1])
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    #plt.suptitle(x)
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def show_t_statistcs(x, save_name=None, recalc_stats=True):
    from post_analysis import calculate_t_test, smooth_data
    slice_indicies = [45, 40, 35, 30]
    fig = plt.figure(figsize=(len(experiment_names)*2 + 1, len(slice_indicies)*1.6))
    axis = fig.subplots(nrows=len(slice_indicies), ncols=len(experiment_names))
    vmin = 2.5
    vmax = 5
    mni = nib.load(mni_image)
    mni_data = mni.get_fdata()

    for idx, exp_name in enumerate(experiment_names):
        exp_dir = root_exp_dir + '/' + exp_name[0]
        if recalc_stats:
            smooth_data(exp_dir, 6)
            calculate_t_test(exp_dir)
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
    #experiment_names = [('nobloodnofull_tv1', 'NB/A'), ('nofull_tv1', 'B/A'), ('noblood_tv1', 'NB/F'), ('data_norm_multi', 'B/F')]
    experiment_names = [('o3_60_tv03_mvg_rs05_nb_nf', 'A1'),
                        ('o3_60_tv03_mvg_rs05_nf', 'A2'),
                        ('o3_60_tv03_mvg_rs05_nb', 'F1'),
                        ('o3_60_tv03_mvg_rs05', 'F2')]
    # show_example_slices_x('dbv', True, save_name='multi_single_examples.pdf')
    compare_experiments_masked_elbo(save_name='forwardmodels_elbo.pdf')

    experiment_names = [('o3_60_tv03_mvg_rs05_nb', 'F1'),
                        ('o3_60_tv03_mvg_rs05', 'F2')]

    compare_experiments_baseline_all(save_name='forwardmodels_stats.pdf')

    #experiment_names = [('noblood_tv1', 'NB/F'),  ('data_norm_multi', 'B/F')]
    #compare_conditions_all(experiment_names[0], 'nbf_condition_dist.pdf')
    #compare_conditions_all(experiment_names[1], 'bf_condition_dist.pdf')

    #show_t_statistcs('oef', save_name='forwardmodels_tstat_oef.pdf', recalc_stats=False)
    #show_t_statistcs('dbv', save_name='forwardmodels_tstat_dbv.pdf', recalc_stats=False)
    #show_t_statistcs('r2p', save_name='forwardmodels_tstat_r2p.pdf', recalc_stats=False)
#compare_x_conditions('r2p', root_exp_dir + '/' + experiment_names[0])
# Smoothness
if False:
    experiment_names = [('o3_60_tv0_mvg_rs05', 'TV 0'),
                        ('o3_60_tv01_mvg_rs05', 'TV 0.1'),
                        ('o3_60_tv02_mvg_rs05', 'TV 0.2'),
                        ('o3_60_tv03_mvg_rs05', 'TV 0.3'),
                        ('o3_60_tv04_mvg_rs05', 'TV 0.4'),
                        ('o3_60_tv05_mvg_rs05', 'TV 0.5')]
    #show_t_statistcs('oef', save_name='smoothness_tstat_oef.pdf', recalc_stats=False)
    #show_t_statistcs('dbv', save_name='smoothness_tstat_dbv.pdf', recalc_stats=False)
    #show_t_statistcs('r2p', save_name='smoothness_tstat_r2p.pdf', recalc_stats=False)

    show_example_slices_x('dbv', True, save_name='smoothness_examples_dbv.pdf', last_two=True)
    show_example_slices_x('oef', True, save_name='smoothness_examples_oef.pdf', last_two=True)

    experiment_names = [('o3_60_tv0_mvg_rs05', '0'),
                        ('o3_60_tv01_mvg_rs05', '0.1'),
                        ('o3_60_tv02_mvg_rs05', '0.2'),
                        ('o3_60_tv03_mvg_rs05', '0.3'),
                        ('o3_60_tv04_mvg_rs05', '0.4'),
                        ('o3_60_tv05_mvg_rs05', '0.5')]
    compare_experiments_baseline_all(save_name='smoothness_stats.pdf', plot_line=True, x_label='Total Variation')
    #show_example_slices_x('r2p', True, save_name='smoothness_examples_r2p.pdf')

if False:
    create_pt_exp(root_exp_dir + '/' + 'o3_60_tv01_mvg_rs05_synthu1_g4', root_exp_dir + '/' + 'o3_60_tv01_mvg_rs05_synthu1_g4_pt')
    experiment_names = [('wls_clip', 'WLS'),
                        ('o3_60_tv05_mvg_rs05_pt', 'Synth'),
                        ('o3_60_tv01_mvg_rs05', 'VI'),
                        ('o3_60_tv01_mvg_rs05_synthu1_g4_pt', 'Synth R'),
                        ('o3_60_tv01_mvg_rs05_synthu1_g4', 'VI R'),
                        ('o3_60_tv03_mvg_rs05_synthu1_gz4_ws3_uf09', 'VI R 2'),
                        ('o3_60_tv03_mvg_rs05_synthu1_g4_ws3_uf09_ub', "UB")]
    # , ('optimal3_tv1_newblood2v2_ms025', 'VI + TV')]
    compare_experiments_baseline_all(save_name='restricted.pdf')
    compare_conditions_all(experiment_names[-1], 'restricted_condition_dist.pdf')

    show_example_slices_x('dbv', False, save_name='restricted_dbv.pdf')
    show_example_slices_x('oef', False, save_name='restricted_oef.pdf')
    show_example_slices_x('r2p', False, save_name='restricted_r2p.pdf')

    show_t_statistcs('oef', save_name='restricted_tstat_oef.pdf', recalc_stats=True)
    show_t_statistcs('dbv', save_name='restricted_tstat_dbv.pdf', recalc_stats=False)
    show_t_statistcs('r2p', save_name='restricted_tstat_r2p.pdf', recalc_stats=False)


if False:
    create_pt_exp(root_exp_dir+'/'+'o3_60_tv05_mvg_rs05', root_exp_dir+'/'+'o3_60_tv05_mvg_rs05_pt')
    experiment_names = [('wls_clip', 'WLS'),
                        ('optimal3_tv0_newblood2_pt', 'Synth'),
                        ('optimal3_tv0_newblood2', 'VI'),
                        ('optimal3_tv1_newblood2v2_ms05', 'VI + TV')]

    experiment_names = [('wls_clip', 'WLS'),
                        ('o3_60_tv05_mvg_rs05_pt', 'Synth'),
                        ('o3_60_tv0_mvg_rs05', 'VI'),
                        ('o3_60_tv03_mvg_rs05', 'VI + TV')]
                        #, ('optimal3_tv1_newblood2v2_ms025', 'VI + TV')]
    compare_experiments_baseline_all(save_name='inference_stats.pdf')

    show_example_slices_x('dbv', True, save_name='vi_syth_wls_examples_dbv.pdf')
    show_example_slices_x('oef', True, save_name='vi_syth_wls_examples_oef.pdf')
    show_example_slices_x('r2p', True, save_name='vi_syth_wls_examples_r2p.pdf')
    show_example_slices_x('dbv', False, save_name='vi_syth_wls_examples_hyperv_dbv.pdf')
    show_example_slices_x('oef', False, save_name='vi_syth_wls_examples_hyperv_oef.pdf')
    show_example_slices_x('r2p', False, save_name='vi_syth_wls_examples_hyperv_r2p.pdf')

    show_t_statistcs('oef', save_name='vi_synth_tstat_oef.pdf', recalc_stats=True)
    show_t_statistcs('dbv', save_name='vi_synth_tstat_dbv.pdf', recalc_stats=False)
    show_t_statistcs('r2p', save_name='vi_synth_tstat_r2p.pdf', recalc_stats=False)
    compare_conditions_all(experiment_names[-1], 'vitv_condition_dist.pdf')

if True:
    import scipy.stats  as ss
    import configparser
    config = configparser.ConfigParser()
    config.read('config')
    params = config['DEFAULT']
    oef_mean = float(params['oef_mean'])
    oef_std = float(params['oef_std'])
    dbv_mean = float(params['dbv_mean'])
    dbv_std = float(params['dbv_std'])

    oef_start = float(params['oef_start'])
    oef_end = float(params['oef_end'])
    dbv_start = float(params['dbv_start'])
    dbv_end = float(params['dbv_end'])

    fig = plt.figure(figsize=(6, 5))
    axis = fig.subplots(nrows=1, ncols=2)

    def plot_pdf(_ax, _mean, _std, _start, _end, _u_prop):
        x = np.linspace(_start, _end, 100)
        norm_pdf = ss.norm(loc=_mean, scale=_std).pdf(x) * (1.0-_u_prop)
        uniform_pdf = ss.uniform(loc=_start, scale=_end-_start).pdf(x) * _u_prop
        _ax.plot(x, (norm_pdf+uniform_pdf), label="uniform: " + str(_u_prop))
        _ax.set_ylim(0, 4.0 / (_end-_start))


    for prop in (0.0, 0.25, 0.5, 0.75, 1.0):
        plot_pdf(axis[0], oef_mean, oef_std, oef_start, oef_end, prop)
        plot_pdf(axis[1], dbv_mean, dbv_std, dbv_start, dbv_end, prop)
    axis[0].set_xlabel('oef')
    axis[0].set_ylabel('probability')
    axis[1].set_xlabel('dbv')

    fig.tight_layout()
    #fig.subplots_adjust(right=0.88)
    #cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    axis[1].legend()

    plt.savefig('prior_distributions.pdf')
if True:
    experiment_names = [('prior_exp_o3_40_tv0_mvg_rs05_synthu_0_g4_ws3_uf09', '0'),
                        ('prior_exp_o2_20_tv03_mvg_rs05_synthu_025_g4_bb_l2mlp1e3', '0.25'),
                        ('prior_exp_o3_40_tv0_mvg_rs05_synthu_05_g4_ws3_uf09', '0.5'),
                        ('prior_exp_o3_40_tv0_mvg_rs05_synthu_075_g4_ws3_uf09', '0.75'),
                        ('prior_exp_o3_40_tv03_mvg_rs05_synthu_1_g4_ws3_uf09', '1.0')]


    experiment_names = [('wls_clip', 'WLS'),
                        ('prior_exp_o3_40_tv03_mvg_rs05_synthu_0_g4_ws3_uf09', '0'),
                        ('prior_exp_o3_40_tv03_mvg_rs05_synthu_025_g4_ws3_uf09', '0.25'),
                        ('prior_exp_o3_40_tv03_mvg_rs05_synthu_05_g4_ws3_uf09', '0.5'),
                        ('prior_exp_o3_40_tv03_mvg_rs05_synthu_075_g4_ws3_uf09', '0.75'),
                        ('prior_exp_o3_40_tv03_mvg_rs05_synthu_1_g4_ws3_uf09', '1.0')]

    experiment_names = [('prior_exp_o2_60_tv0_mvg_rs05_synthu_0_g4_bb64_swa_lrr95_adamw1e4v2_smfix', '0.0'),
                        ('prior_exp_o2_60_tv1_mvg_rs05_synthu_0_g4_bb64_swa_lrr95_adamw1e4v2_smfix', '0.0s'),
                        ('prior_exp_o2_60_tv03_mvg_rs05_synthu_025_g4_bb64_swa_lrr95_adamw1e4v2', '0.25'),
                        ('prior_exp_o2_60_tv1_mvg_rs05_synthu_025_g4_bb64_swa_lrr95_adamw1e4v2_smfix', '0.25s'),
                        ('prior_exp_o2_60_tv03_mvg_rs05_synthu_05_g4_bb64_l2all1e3_lr2e3', '0.5'),
                        ('prior_exp_o2_60_tv03_mvg_rs05_synthu_075_g4_bb64_l2all1e3_lr2e3', '0.75'),
                        ('prior_exp_o2_60_tv03_mvg_rs05_synthu_1_g4_bb64_swa_lrr95_adamw1e4', '1.0'),
                        ('prior_exp_o2_60_tv1_mvg_rs05_synthu_1_g4_bb64_swa_lrr95_adamw1e4v2_smfix', '1.0s'),
                        ('prior_exp_o2_60_tv2_mvg_rs05_synthu_1_g4_bb64_swa_lrr95_adamw1e4v2_smfix', '1.0s')]
    compare_experiments_masked_elbo(save_name='prior_elbo.pdf')
    compare_experiments_baseline_all(save_name='prior_dist.pdf')


    show_example_slices_x('dbv', True, save_name='prior_examples_dbv.pdf')
    show_example_slices_x('oef', True, save_name='prior_examples_oef.pdf')

    show_t_statistcs('oef', save_name='prior_tstat_oef.pdf', recalc_stats=True)
    show_t_statistcs('dbv', save_name='prior_tstat_dbv.pdf', recalc_stats=False)
    show_t_statistcs('r2p', save_name='prior_tstat_r2p.pdf', recalc_stats=False)

