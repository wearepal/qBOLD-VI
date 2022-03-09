# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Create the figures for the paper

import sys, os, argparse, yaml
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

oef_symbol = '$OEF$'
dbv_symbol = '$\mu$'
from train import setup_argparser, get_defaults

root_exp_dir = '/Users/is321/Documents/Data/qBold/model_predictions/'


baseline_gm_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/baseline_ase_gm.nii.gz'
hyperv_gm_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/hyperv_ase_gm.nii.gz'
baseline_tmean = '/Users/is321/Documents/Data/qBold/hyperv_data/baseline_ase_tmean.nii.gz'
hyperv_tmean = '/Users/is321/Documents/Data/qBold/hyperv_data/hyperv_ase_tmean.nii.gz'
mni_image = '/Users/is321/Documents/Data/qBold/hyperv_data/MNI152_t1_2mm.nii.gz'
baseline_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/baseline_ase_mask.nii.gz'
hyperv_masks = '/Users/is321/Documents/Data/qBold/hyperv_data/hyperv_ase_mask.nii.gz'

subj_masks = [0, 1, 2, 4, 5]

def get_mask_data(baseline, gm, select_images=True):
    if baseline:
        if gm:
            mask = nib.load(baseline_gm_masks)
        else:
            mask = nib.load(baseline_masks)
    else:
        if gm:
            mask = nib.load(hyperv_gm_masks)
        else:
            mask = nib.load(hyperv_masks)
    mask_data = mask.get_fdata()
    if select_images:
        mask_data = mask_data[:, :, :, subj_masks]
    return mask_data

def get_data_x(x, exp_directory, baseline=True, pt=False, select_images=True):

    if x == 'elbo':
        likelihood = get_data_x('likelihood', exp_directory, baseline=baseline, select_images=select_images)
        kl = get_data_x('kl', exp_directory, baseline=baseline, select_images=select_images)
        return (likelihood + kl) * -1.0

    if x == 'oef_std' or x == 'dbv_std':
        data = get_data_x('logstds', exp_directory, baseline=baseline, select_images=False)
        if x == 'oef_std':
            data = data[:, :, :, np.arange(0, 3*6, 3)]
        elif x == 'dbv_std':
            data = data[:, :, :, np.arange(1, 3*6, 3)]
        if select_images:
            data = data[:, :, :, subj_masks]
        return data
    else:
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
        if select_images:
            data = data[:, :, :, subj_masks]
        return data

def get_tstat_x(x, exp_directory, pt=False):
    assert pt==False
    filename = exp_directory + '/' + x + '_mni_t_test.nii.gz'
    img = nib.load(filename)
    data = img.get_fdata()
    return data

def get_masked_x(x, exp_directory, baseline=True, pt=False, gm=True):
    data = get_data_x(x, exp_directory, baseline, pt)

    mask = get_mask_data(baseline, gm)

    return data.flatten()[mask.flatten() == 1.0]

def get_mean_std_masked_x(x, exp_directory, baseline=True, pt=False, gm=True):
    data = get_data_x(x, exp_directory, baseline, pt)
    mask = get_mask_data(baseline, gm)

    means = []
    stds = []
    for idx in range(data.shape[3]):
        masked_data = (data[:,:, :, idx]).flatten()[mask[:,:, :, idx].flatten() == 1.0]
        means.append(masked_data.mean())
        stds.append(np.std(masked_data))
    print(means)
    return means, stds

def get_masked_elbo(exp_directory, baseline=True, gm=True):
    likelihood = get_masked_x('likelihood', exp_directory, baseline=baseline, gm=gm)
    kl = get_masked_x('kl', exp_directory, baseline=baseline, gm=gm)
    return likelihood+kl

def get_masked_elbo_both(exp_directory, gm):
    return np.concatenate([get_masked_elbo(exp_directory, True, gm=gm), get_masked_elbo(exp_directory, False, gm=gm)])

def set_violin_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    labels = [x[1] for x in labels]
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def compare_x_conditions(x, exp_directory, gm=True):
    baseline_x = get_masked_x(x, exp_directory, True, gm=gm)
    hyperv_x = get_masked_x(x, exp_directory, False, gm=gm)

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
        multiplier = 100.
        if x == 'r2p':
            multiplier = 1.0
        format_string = "{:.1f}"
        if x == 'dbv':
            format_string = "{:.2f}"
        print(exp_tuple[1], x, format_string.format(np.mean(data[0])*multiplier), '\pm',
              format_string.format(np.std(data[0])*multiplier),  ' & ',
              format_string.format(np.mean(data[1]) * multiplier), '\pm',
              format_string.format(np.std(data[1]) * multiplier))
    plt.suptitle(exp_tuple[1])
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def compare_experiments_baseline_all(save_name=None, plot_line=False, xs=('oef', 'dbv', 'r2p'), x_label=None):
    fig, axis = plt.subplots(nrows=1, ncols=len(xs), figsize=(9, 4))
    for idx, x in enumerate(xs):
        if len(xs) == 1:
            ax = axis
        else:
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

def get_plot_range(_x, old_range):
    if _x == 'oef':
        if old_range:
            vmin = 0.1
            vmax = 0.4
        else:
            vmin = 0.1 # 0.2
            vmax = 0.5 # 0.5
    elif _x == 'dbv':
        if old_range:
            vmin = 0.04
            vmax = 0.2
        else:
            vmin = 0.01
            vmax = 0.1 # 0.075
    elif _x == 'r2p':
        if old_range:
            vmin = 0.0
            vmax = 10
        else:
            vmin = 0.0
            vmax = 10
    elif _x == 'elbo':
        vmin = 10.0
        vmax = 30.0

    elif _x == 'oef_std':
        vmin = 5e-4
        vmax = 1.5e-2
    elif _x == 'dbv_std':
        vmin = 0.0
        vmax = 2e-4
    return vmin, vmax

def get_im_range(im):
    if im.shape[1] > 64:
        min_x, max_x, min_y, max_y = (15, -15, 10, -10)
    else:
        min_x, max_x, min_y, max_y = (10, -10, 5, -5)
    return min_x, max_x, min_y, max_y

def create_x_maps_conditions(exp_name, _x, save_name=None, old_range=False):
    subj_indicies = [3, 5]
    slice_indicies = [5, 6]

    tmean_images = get_mean_signal_image(True, False)
    masks = get_masks(True, False)
    fig = plt.figure(figsize=((2 + 1) * 1.6 + 1, len(subj_indicies) * 2))
    axis = fig.subplots(nrows=len(subj_indicies), ncols=2 + 1)
    vmin, vmax = get_plot_range(_x, old_range)

    min_x, max_x, min_y, max_y = get_im_range(tmean_images)

    for slice_idx in range(len(subj_indicies)):
        ax = axis[slice_idx, 0]
        im = ax.imshow(transform_im_to_plot(tmean_images[min_x:max_x, min_y:max_y, slice_indicies[slice_idx], subj_indicies[slice_idx]]), cmap='gray')
        ax.axis('off')

    for idx, baseline in enumerate((True, False)):
        exp_dir = root_exp_dir + '/' + exp_name[0]
        data = get_data_x(_x, exp_dir, baseline, select_images=False)
        data = data * masks

        for slice_idx in range(len(subj_indicies)):
            ax = axis[slice_idx, idx + 1]
            alpha = np.float32(transform_im_to_plot(data[min_x:max_x, min_y:max_y, slice_indicies[slice_idx], subj_indicies[slice_idx]] != 0.0))
            im = ax.imshow(transform_im_to_plot(data[min_x:max_x, min_y:max_y, slice_indicies[slice_idx], subj_indicies[slice_idx]]), vmin=vmin,
                           vmax=vmax, alpha=alpha)
            ax.axis('off')
            if slice_idx == 0 and idx == 0:
                ax.set_title('Baseline')
            elif slice_idx == 0 and idx == 1:
                ax.set_title('Hyperventilation')
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_alpha(1)
    cbar.draw_all()
    # plt.suptitle(x)
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def elbo_plots(_x='elbo', gm=True, save_name=None):

    baseline_means = []
    hyperv_means = []
    baseline_stds = []
    hyperv_stds = []
    labels = [x[1] for x in experiment_names]

    for idx, exp in enumerate(experiment_names):
        exp_means_baseline, std = get_mean_std_masked_x(_x, root_exp_dir + '/' + exp[0], baseline=True, gm=gm)
        exp_means_hyperv, std = get_mean_std_masked_x(_x, root_exp_dir + '/' + exp[0], baseline=False, gm=gm)

        baseline_means.append(np.round(np.mean(exp_means_baseline), 1))
        hyperv_means.append(np.round(np.mean(exp_means_hyperv), 1))
        baseline_stds.append(np.std(exp_means_baseline))
        hyperv_stds.append(np.std(exp_means_hyperv))

        print(labels[idx], baseline_means[-1], baseline_stds[-1], hyperv_means[-1], hyperv_stds[-1])

    x_val = np.arange(len(experiment_names))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(len(experiment_names)*1.25+1, 5))
    rects1 = ax.bar(x_val - width / 2, baseline_means, width, label='Baseline', yerr=baseline_stds)
    rects2 = ax.bar(x_val + width / 2, hyperv_means, width, label='Hyperventilation', yerr=hyperv_stds)

    min_val = min(np.min(baseline_means), np.min(hyperv_means))
    max_val = max(np.max(baseline_means), np.max(hyperv_means))
    if _x == 'elbo':
        ax.set_ylim(18, 28)
        ax.set_ylabel('ELBO')
    else:
        ax.set_ylim(max(min_val - (max_val-min_val)*0.5, 0.0), max_val + (max_val-min_val)*0.3)
        ax.set_ylabel(_x)


    ax.set_xticks(x_val)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(0.9, 1.13))

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    #fig.tight_layout()

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()



def compare_experiments_masked_elbo(save_name=None, plot_line=True, gm=True):
    elbos = []
    medians = []

    print('masked elbo')
    for exp in experiment_names:
        baseline_elbo = get_masked_elbo(root_exp_dir + '/' + exp[0], baseline=True, gm=gm)
        hyperv_elbo = get_masked_elbo(root_exp_dir + '/' + exp[0], baseline=False, gm=gm)
        comb_elbo = np.concatenate([baseline_elbo, hyperv_elbo])
        elbos.append(comb_elbo)
        medians.append(np.median(elbos[-1]))
        print(exp[1], np.mean(baseline_elbo), np.std(baseline_elbo), np.mean(hyperv_elbo), np.std(hyperv_elbo),
              np.mean(comb_elbo), np.std(comb_elbo))

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(len(experiment_names)*2+1, 4))
    ax1.violinplot(elbos, showmedians=True, showextrema=False)

    set_violin_axis_style(ax1, experiment_names)
    if gm:
        ax1.set_ylim(np.min(elbos), np.percentile(elbos, 98))
        ax1.set_ylabel('GM Voxelwise ELBO')
    else:
        ax1.set_ylim(np.percentile(elbos,2), np.percentile(elbos,98))

    if plot_line:
        ax1.plot(1 + np.arange(len(experiment_names)), np.array(medians))

    plt.tight_layout()

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()

def get_masks(baseline, select_subjects=True):
    mask_filename = hyperv_masks
    if baseline:
        mask_filename = baseline_masks

    mask_nib = nib.load(mask_filename)
    masks = mask_nib.get_fdata()
    if select_subjects:
        masks = masks[:, :, :, subj_masks]
    return masks

def get_mean_signal_image(baseline, select_subjects=True):
    im_filename = hyperv_tmean
    if baseline:
        im_filename = baseline_tmean

    im_nib = nib.load(im_filename)
    tmean_images = im_nib.get_fdata()
    if select_subjects:
        tmean_images = tmean_images[:, :, :, subj_masks]

    return tmean_images

def show_example_slices_x(x, baseline, save_name=None, first_two=True, exp_idx_old_range=[],
                          subj_indicies=[0, 1, 4, 2], slice_indicies=[6, 5, 3, 2], show_title=False):

    if first_two:
        subj_indicies = subj_indicies[:2]
        slice_indicies = slice_indicies[:2]

    masks = get_masks(baseline)
    tmean_images = get_mean_signal_image(baseline)

    fig = plt.figure(figsize=((len(experiment_names)+1)*1.6 + 1, len(subj_indicies)*2))
    axis = fig.subplots(nrows=len(subj_indicies), ncols=len(experiment_names)+1)

    min_x, max_x, min_y, max_y = get_im_range(tmean_images)

    for slice_idx in range(len(subj_indicies)):
        ax = axis[slice_idx, 0]
        im = ax.imshow(transform_im_to_plot(tmean_images[min_x:max_x, min_y:max_y, slice_indicies[slice_idx], subj_indicies[slice_idx]]), cmap='gray')
        ax.axis('off')

    for idx, exp_name in enumerate(experiment_names):
        exp_dir = root_exp_dir + '/' + exp_name[0]
        data = get_data_x(x, exp_dir, baseline)
        data = data * masks
        exp_vmin, exp_vmax = get_plot_range(x, idx in exp_idx_old_range)

        for slice_idx in range(len(subj_indicies)):
            ax = axis[slice_idx, idx+1]
            im = ax.imshow(transform_im_to_plot(data[min_x:max_x, min_y:max_y, slice_indicies[slice_idx], subj_indicies[slice_idx]]), vmin=exp_vmin, vmax=exp_vmax)
            ax.axis('off')
            if slice_idx == 0:
                ax.set_title(exp_name[1])
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    title = None
    if x == 'oef':
        title = 'OEF '
    elif x == 'dbv':
        title = 'DBV '
    elif x == 'r2p':
        title = 'R2\' '
    if baseline:
        title = title + 'at baseline.'
    else:
        title = title + 'during hyperventilation.'
    if title and show_title:
        plt.suptitle(title)

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()

def transform_im_to_plot(im_to_plot):
    return np.copy(im_to_plot).transpose()[::-1, :]

def show_combo_example_slices_x(x, save_name=None, first_two=True, exp_idx_old_range=[],
                          subj_indicies=[0, 1, 4, 2], slice_indicies=[6, 5, 3, 2]):

    if first_two:
        subj_indicies = subj_indicies[:2]
        slice_indicies = slice_indicies[:2]


    tmean_images = get_mean_signal_image(True)

    fig = plt.figure(figsize=((len(experiment_names*2)+1)*1.6 + 1, len(subj_indicies)*2))
    axis = fig.subplots(nrows=len(subj_indicies), ncols=len(experiment_names*2)+1)

    min_x, max_x, min_y, max_y = get_im_range(tmean_images)

    for slice_idx in range(len(subj_indicies)):
        ax = axis[slice_idx, 0]
        im = ax.imshow(transform_im_to_plot(tmean_images[min_x:max_x, min_y:max_y, slice_indicies[slice_idx], subj_indicies[slice_idx]]), cmap='gray')
        ax.axis('off')

    a = list()
    for exp in experiment_names:
        a.extend([exp, exp])
    for idx, exp_name in enumerate(a):
        if idx % 2 == 0:
            baseline = True
        else:
            baseline = False
        masks = get_masks(baseline)
        exp_dir = root_exp_dir + '/' + exp_name[0]
        data = get_data_x(x, exp_dir, baseline)
        data = data * masks
        exp_vmin, exp_vmax = get_plot_range(x, idx in exp_idx_old_range)

        for slice_idx in range(len(subj_indicies)):
            ax = axis[slice_idx, idx+1]
            im = ax.imshow(transform_im_to_plot(data[min_x:max_x, min_y:max_y, slice_indicies[slice_idx], subj_indicies[slice_idx]]), vmin=exp_vmin, vmax=exp_vmax)
            ax.axis('off')
            if slice_idx == 0:
                title = exp_name[1]
                if baseline == False:
                    title += '-hyper'
                ax.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    """title = None
    if x == 'oef':
        title = 'OEF '
    elif x == 'dbv':
        title = 'DBV '
    elif x == 'r2p':
        title = 'R2\' '
    if baseline:
        title = title + 'at baseline.'
    else:
        title = title + 'during hyperventilation.'
    if title:
        plt.suptitle(title)"""

    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()


def show_t_statistcs(x, save_name=None, recalc_stats=True, show_title=False):
    from post_analysis import calculate_t_test, smooth_data
    slice_indicies = [50, 45, 40, 35, 30]
    fig = plt.figure(figsize=(len(experiment_names)*1.6 + 1, len(slice_indicies)*2))
    axis = fig.subplots(nrows=len(slice_indicies), ncols=len(experiment_names))
    vmin = 3.5
    vmax = 8
    mni = nib.load(mni_image)
    mni_data = mni.get_fdata()

    for idx, exp_name in enumerate(experiment_names):
        exp_dir = root_exp_dir + '/' + exp_name[0]
        if recalc_stats:
            smooth_data(exp_dir, 6)
            calculate_t_test(exp_dir)
        data = get_tstat_x(x, exp_dir)*-1.0
        for slice_idx in range(len(slice_indicies)):
            ax = axis[slice_idx, idx]
            slice = data[15:-15, 10:-10, slice_indicies[slice_idx], 0]
            mni_slice = mni_data[15:-15, 10:-10, slice_indicies[slice_idx]]

            ax.imshow(transform_im_to_plot(mni_slice), cmap='gray')

            slice_pos_mask = transform_im_to_plot((slice > vmin) * 1.0)
            slice_pos = transform_im_to_plot(slice )* slice_pos_mask
            ax.imshow(slice_pos, alpha=slice_pos_mask, vmin=vmin, vmax=vmax, cmap='hot')
            im = ax.imshow(slice_pos, alpha=0.0, vmin=vmin, vmax=vmax, cmap='hot')

            slice_neg_mask = transform_im_to_plot((slice < -vmin) * 1.0)
            slice_neg = transform_im_to_plot(slice) * slice_neg_mask
            ax.imshow(slice_neg, alpha=slice_neg_mask, vmin=-vmax, vmax=-vmin, cmap='cool')
            im2 = ax.imshow(slice_neg, alpha=0.0, vmin=-vmax, vmax=-vmin, cmap='cool')
            ax.axis('off')
            if slice_idx == 0:
                ax.set_title(exp_name[1])
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.8, 0.55, 0.05, 0.35])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_alpha(1)
        cbar.draw_all()

        cbar_ax = fig.add_axes([0.8, 0.1, 0.05, 0.35])
        cbar = fig.colorbar(im2, cax=cbar_ax)
        cbar.set_alpha(1)
        cbar.draw_all()
        if show_title:
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
    experiment_names = [('prior_exp_o2_60_tv2_mvg_rs05_synthu_0_g4_bb64_swa_lrr95_adamw1e4v2_smfix_mu4_sin', 'single'),
                        ('prior_exp_o2_60_tv2_mvg_rs05_synthu_0_g4_bb64_swa_lrr95_adamw1e4v2_mu4', 'multi')]

    compare_experiments_baseline_all(save_name='multi_single_stats.pdf')
    elbo_plots(save_name='multi_single_elbo_plots.pdf')
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

# Arch
if False:
    #experiment_names = [('nobloodnofull_tv1', 'NB/A'), ('nofull_tv1', 'B/A'), ('noblood_tv1', 'NB/F'), ('data_norm_multi', 'B/F')]
    experiment_names = [('optimal_kl70_mlp_s2_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'MLP'),
                        ('optimal_kl70_s2_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'Conv'),
                        ('optimal_kl70_s2_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9_gba', 'GBA')]
    # show_example_slices_x('dbv', True, save_name='multi_single_examples.pdf')
    elbo_plots(save_name='mlp_elbo_gm.pdf', gm=True)
    elbo_plots(save_name='mlp_elbo.pdf', gm=False)
    compare_experiments_baseline_all(save_name='mlp_stats.pdf')
    show_example_slices_x('oef', True, save_name='mlp_examples_oef.pdf')
    exit()

# Forward model experiments
if True:
    #experiment_names = [('nobloodnofull_tv1', 'NB/A'), ('nofull_tv1', 'B/A'), ('noblood_tv1', 'NB/F'), ('data_norm_multi', 'B/F')]
    experiment_names = [('optimal_nb_nf_ptadamw50e4', 'A1'),
                        ('optimal_nf_ptadamw50e4', 'A2'),
                        ('optimal_nb', 'F1'),
                        ('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'F2')]
    # show_example_slices_x('dbv', True, save_name='multi_single_examples.pdf')
    compare_experiments_masked_elbo(save_name='forwardmodels_elbo.pdf')
    elbo_plots(save_name='forward_models_elbo_gm.pdf', gm=True)
    elbo_plots(save_name='forward_models_elbo.pdf', gm=False)
    experiment_names = [experiment_names[-2], experiment_names[-1]]

    compare_experiments_baseline_all(save_name='forwardmodels_stats.pdf')

    #experiment_names = [('noblood_tv1', 'NB/F'),  ('data_norm_multi', 'B/F')]
    #compare_conditions_all(experiment_names[0], 'nbf_condition_dist.pdf')
    #compare_conditions_all(experiment_names[1], 'bf_condition_dist.pdf')

    #show_t_statistcs('oef', save_name='forwardmodels_tstat_oef.pdf', recalc_stats=False)
    #show_t_statistcs('dbv', save_name='forwardmodels_tstat_dbv.pdf', recalc_stats=False)
    #show_t_statistcs('r2p', save_name='forwardmodels_tstat_r2p.pdf', recalc_stats=False)
#compare_x_conditions('r2p', root_exp_dir + '/' + experiment_names[0])
# Smoothness
if True:
    experiment_names = [('o3_60_tv0_mvg_rs05', 'TV 0'),
                        ('o3_60_tv01_mvg_rs05', 'TV 0.1'),
                        ('o3_60_tv02_mvg_rs05', 'TV 0.2'),
                        ('o3_60_tv03_mvg_rs05', 'TV 0.3'),
                        ('o3_60_tv04_mvg_rs05', 'TV 0.4'),
                        ('o3_60_tv05_mvg_rs05', 'TV 0.5')]
    #


    experiment_names = [('optimal_kl70_s2_tv0_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '0'),
                        ('optimal_kl70_s2_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '2.5'),
                        ('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '5'),
                        ('optimal_kl70_s2_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '7.5'),
                        ('optimal_kl70_s2_tv10_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '10')]
    show_example_slices_x('dbv', True, save_name='smoothness_examples_dbv.pdf', first_two=True)
    show_example_slices_x('oef', True, save_name='smoothness_examples_oef.pdf', first_two=True)
    compare_experiments_baseline_all(save_name='smoothness_stats.pdf', plot_line=True, x_label='Total Variation')
    elbo_plots('elbo', save_name='smooth_elbo_gm.pdf')
    elbo_plots('elbo', gm=False, save_name='smooth_elbo_allvox.pdf')
    if False:
        show_t_statistcs('oef', save_name='smoothness_tstat_oef.pdf', recalc_stats=True)
        show_t_statistcs('dbv', save_name='smoothness_tstat_dbv.pdf', recalc_stats=False)
        show_t_statistcs('r2p', save_name='smoothness_tstat_r2p.pdf', recalc_stats=False)
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


if True:
    create_pt_exp(root_exp_dir+'/'+'optimal_kl70_s2_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', root_exp_dir+'/'+'optimal_kl70_s2_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9_pt')
    experiment_names = [('wls_clip', 'WLS'),
                        ('optimal_kl70_s2_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9_pt', 'Synth'),
                        ('optimal_tv0', 'VI'),
                        ('optimal_kl70_s2_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'VI+TV')]

    experiment_names = [('wls_clip', 'WLS'),
                        ('optimal_kl70_s2_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9_pt', 'Synth'),
                        ('optimal_kl70_s2_tv0_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'VI'),
                        ('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'VI+TV')]

    compare_conditions_all(experiment_names[0], 'wls_condition_dist.pdf')
    compare_conditions_all(experiment_names[-1], 'vitv_condition_dist.pdf')
    compare_conditions_all(experiment_names[1], 'synth_condition_dist.pdf')
    compare_conditions_all(experiment_names[2], 'vi_condition_dist.pdf')
    exit()
    compare_experiments_baseline_all(save_name='inference_stats.pdf')
    experiment_names = [('wls_clip', 'WLS'),
                        (
                        'optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'VI+TV')]
    show_combo_example_slices_x('dbv', save_name='vi_syth_wls_examples_dbv_combo.pdf', exp_idx_old_range=[])
    show_combo_example_slices_x('oef',  save_name='vi_syth_wls_examples_oef_combo.pdf', exp_idx_old_range=[])
    show_combo_example_slices_x('r2p', save_name='vi_syth_wls_examples_r2p_combo.pdf', exp_idx_old_range=[])
    if True:
        show_t_statistcs('oef', save_name='vi_synth_tstat_oef.pdf', recalc_stats=False)
        show_t_statistcs('dbv', save_name='vi_synth_tstat_dbv.pdf', recalc_stats=False)
        show_t_statistcs('r2p', save_name='vi_synth_tstat_r2p.pdf', recalc_stats=False)

if False:
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

    def plot_pdf(_ax, _mean, _std, _start, _end, _u_prop, label, y_max):
        x = np.linspace(_start, _end, 100)
        _a, _b = (_start - _mean) / _std, (_end - _mean) / _std
        norm_pdf = ss.truncnorm(a=_a, b=_b, loc=_mean, scale=_std).pdf(x) * (1.0-_u_prop)
        uniform_pdf = ss.uniform(loc=_start, scale=_end-_start).pdf(x) * _u_prop
        _ax.plot(x, (norm_pdf+uniform_pdf), label=label)
        _ax.set_ylim(0, y_max)

    oef_max = 5
    dbv_max = 40

    plot_pdf(axis[0], 0.4, 0.1, oef_start, oef_end, 1.0, '$\mathcal{U}$', oef_max)
    plot_pdf(axis[1], 0.025, 0.01, dbv_start, dbv_end, 1.0, '$\mathcal{U}$', dbv_max)

    plot_pdf(axis[0], 0.4, 0.3, oef_start, oef_end, 0.0, '$\mathcal{N}$w', oef_max)
    plot_pdf(axis[1], 0.025, 0.03, dbv_start, dbv_end, 0.0, '$\mathcal{N}$w', dbv_max)

    plot_pdf(axis[0], 0.4, 0.1, oef_start, oef_end, 0.0, '$\mathcal{N}$n', oef_max)
    plot_pdf(axis[1], 0.025, 0.01, dbv_start, dbv_end, 0.0, '$\mathcal{N}$n', dbv_max)

    plot_pdf(axis[0], 0.4, 0.2, oef_start, oef_end, 0.0, '$\mathcal{N}$', oef_max)
    plot_pdf(axis[1], 0.025, 0.02, dbv_start, dbv_end, 0.0, '$\mathcal{N}$', dbv_max)


    axis[0].set_xlabel('oef')
    axis[0].set_ylabel('probability')
    axis[1].set_xlabel('dbv')

    fig.tight_layout()
    #fig.subplots_adjust(right=0.88)
    #cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    axis[1].legend()

    plt.savefig('prior_distributions.pdf')

if True:
    experiment_names = [('optimal_kl70_diag_s2_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'Diagonal $\Sigma_l$'),
                        ('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'Full $\Sigma_l$')]

    elbo_plots('elbo', save_name='diag_dist_gm.pdf')
    elbo_plots('elbo', gm=False, save_name='diag_dist_allvox.pdf')
    compare_experiments_baseline_all(xs=('oef', 'dbv'), save_name='diag_dist.pdf')

# Uncertainty
if True:
    create_x_maps_conditions(
        ('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'optimal'), 'dbv_std',
        save_name='dbv_std.pdf')
    create_x_maps_conditions(('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'optimal'), 'oef_std', save_name='oef_std.pdf')
    create_x_maps_conditions(('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'optimal'), 'elbo', save_name='elbo_outlier.pdf')

if True:
    experiment_names = [('optimal_mu4_2_dbv25_2', '$\mathcal{N}(0.4, 0.2)$'),
                        ('optimal_mu4_15_dbv25_15', 'b'),
                        ('optimal_mu4_2_dbv25_2', 'a'),
                        ('optimal_mu4_25_dbv25_25_tanh6', 'b'),
                        ('optimal_mu4_25_dbv25_25', 'b'),
                        ('optimal_mu4_3_dbv25_3', 'c'),
                        ('optimal_uniform', '$\mathcal{U}(0.05, 0.8)$')]

    experiment_names = [('optimal_kl70_u_tv75_lossfix_eps1e4_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '$\mathcal{U}$'),
                        ('optimal_kl70_s3_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '$\mathcal{N}$w'),
                        ('optimal_kl70_s1_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '$\mathcal{N}$n'),
                        ('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', '$\mathcal{N}$')]
    [('optimal_kl70_s3_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'new wide'),
    ('optimal_kl70_s2_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'new'),
    ('optimal_kl70_s2_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_nb_adamw2e4_beta2_9', 'nb')]
    #create_x_maps_conditions(('optimal_kl70_lossfix', 'optimal'), 'oef_std')
    #create_x_maps_conditions(('optimal_kl70_lossfix', 'optimal'), 'elbo')

    elbo_plots('elbo', save_name='elbo_dist_gm.pdf')
    elbo_plots('elbo', gm=False, save_name='elbo_dist_allvox.pdf')
    compare_experiments_baseline_all(save_name='prior_dist.pdf', xs=('oef', 'dbv'))
    compare_experiments_baseline_all(save_name='prior_dist_r2p.pdf', xs=('r2p',))

    compare_experiments_masked_elbo(save_name='prior_elbo.pdf')
    compare_experiments_masked_elbo(save_name='prior_elbo_all.pdf', gm=False)
    experiment_names = [('wls_clip', 'WLS'),
                        ('optimal_kl70_u_tv75_lossfix_eps1e4_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9',
                         '$\mathcal{U}$'),
                        ('optimal_kl70_s3_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9',
                         '$\mathcal{N}$w'),
                        ('optimal_kl70_s1_tv75_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9',
                         '$\mathcal{N}$n'),
                        ('optimal_kl70_s2_tv5_lossfix_eps1e6_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9',
                         '$\mathcal{N}$')
                        ]
    compare_experiments_baseline_all(save_name='prior_dist.pdf')

    experiment_names=[('wls_clip', 'WLS'), ('optimal_kl70_u_tv75_lossfix_eps1e4_diagrange3offdiagexp2_newwhiten_adamw2e4_beta2_9', 'VI-$\mathcal{U}$-TV')]
    show_example_slices_x('dbv', True, save_name='prior_examples_dbv.pdf', exp_idx_old_range=[0,1], last_two=True)
    show_example_slices_x('oef', True, save_name='prior_examples_oef.pdf', exp_idx_old_range=[0, 1], last_two=True)
    show_example_slices_x('r2p', True, save_name='prior_examples_r2p.pdf', exp_idx_old_range=[0, 1], last_two=True)
    if False:
        show_t_statistcs('oef', save_name='prior_tstat_oef.pdf', recalc_stats=True)
        show_t_statistcs('dbv', save_name='prior_tstat_dbv.pdf', recalc_stats=False)
        show_t_statistcs('r2p', save_name='prior_tstat_r2p.pdf', recalc_stats=False)

if False:
    baseline_tmean = '/Users/is321/Documents/Data/qBold/streamlined-qBOLD_study/streamlined_ase_tmean.nii.gz'
    baseline_masks = '/Users/is321/Documents/Data/qBold/streamlined-qBOLD_study/streamlined_ase_mask.nii.gz'

    experiment_names = [('wls_clip_streamlined', 'WLS'), ('transfer_notransfer', 'VI+TV')]#, ('transfer_fulllr', 'VI+TV+Transfer')]
    subj_indicies = [6, 6, 6, 6, 6]
    slice_indicies = [8, 6, 4, 2, 1]
    subj_masks = [0, 1, 2, 3, 4, 5, 6]
    show_example_slices_x('dbv', True, save_name='transfer_examples_dbv.pdf', exp_idx_old_range=[], last_two=False,
                          subj_indicies=subj_indicies, slice_indicies=slice_indicies)
    show_example_slices_x('oef', True, save_name='transfer_examples_oef.pdf', exp_idx_old_range=[], last_two=False,
                          subj_indicies=subj_indicies, slice_indicies=slice_indicies)
    show_example_slices_x('r2p', True, save_name='transfer_examples_r2p.pdf', exp_idx_old_range=[], last_two=False,
                          subj_indicies=subj_indicies, slice_indicies=slice_indicies)
