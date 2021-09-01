# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Code for preparing real data for further analysis
import nibabel as nib
import numpy as np
from glob import glob
from os import system
data_dir = '/Users/is321/Documents/Data/qBold/hyperv_data/'


def estimate_noise_level():
    import matplotlib.pyplot as plt

    subjects = ['CISC17352', 'CISC17543', 'CISC17987', 'CISC19890', 'CISC19950', 'CISC20384']
    basenames = ['baseline_ase', 'hyperv_ase']
    norm_snr_list = []
    for subject in subjects:
        for basename in basenames:
            dir_name = data_dir + subject
            brain_mask = dir_name + '/mask_' + basename + '_mask.nii.gz'
            img = nib.load(dir_name + '/' + basename + '.nii.gz')
            data = img.get_fdata()

            mask = nib.load(brain_mask)
            mask = mask.get_fdata()

            mask = mask.reshape(-1)

            within_mask_data = data.reshape(-1, 11)[mask > 0, :]
            outside_mask_data = data.reshape(-1, 11)[mask == 0, :]

            within_mask_mean = np.mean(within_mask_data, 0)
            outside_mask_std = np.std(outside_mask_data, 0)

            # Estimate the SNR from the corner of the image (0 true signal)
            corner_vals = 12
            corner = data[1:corner_vals, 1:corner_vals, :, :]
            corner2 = data[-corner_vals:-1, -corner_vals:-1, :, :]
            corner3 = data[-corner_vals:-1, 1:corner_vals, :, :]
            corner4 = data[1:corner_vals, -corner_vals:-1, :, :]
            corner = np.concatenate([corner, corner2, corner3, corner4], 0)
            # Reshape
            corner = corner.reshape(-1, 11)

            corner_std = np.std(corner[:, :], 0)
            snr = within_mask_mean / corner_std
            norm_snr = snr / snr[3]
            norm_snr_list.append(norm_snr)
            """print(norm_snr, snr)
            plt.hist(data[:,:,:,2].flatten())
            #plt.show()
            # Normalise by tau =0 image and take the log
            corner = np.log(corner / corner[:, 2:3])
            # Log at the std-dev
            print(np.std(corner[:, 3:]))
            # plot the histogram for different taus
            plt.hist(corner)
        
            # Look at the whole image histogram
            whole_image = data.reshape(-1, 11) / data.reshape(-1, 11)[:, 2:3]
            plt.hist(whole_image)"""

    norm_snrs = np.array(norm_snr_list)
    print(np.mean(norm_snrs, 0), np.std(norm_snrs, 0))


def register_to_t1(image_filename):
    # Rigidly register the qbold data to the T1
    #
    import os.path as path

    import subprocess

    print(image_filename)

    dir_name = path.dirname(image_filename)
    basename = path.basename(image_filename).split('.')[0]

    mean_image = dir_name + '/tmean_' + basename + '.nii.gz'
    warped_mean_image = dir_name + '/tmean_warped' + basename + '.nii.gz'
    brain_mask = dir_name + '/mask_' + basename + '_mask.nii.gz'
    T1 = dir_name + '/T1.nii'
    T1_2mm = dir_name + '/T1_2mm.nii.gz'
    T1_2mm_mask = dir_name + '/T1_2mm_mask.nii.gz'
    T1_2mm_invmask = dir_name + '/T1_2mm_invmask.nii.gz'
    transform_matrix = dir_name + '/' + basename + 'toT1.mat'
    transform_matrix_inv = dir_name + '/' + 'T1to' + basename + '.mat'
    transform_2_roi = dir_name + '/' + basename + 'toT1_roi.mat'
    warp_to_std = dir_name + '/' + basename + '_warp_to_std.nii.gz'
    anat_dir = dir_name + '/T1_2mm.anat/'
    seg_wm_gm_out = dir_name + '/' + basename + 'wm_gm'
    t1_to_ase_field = dir_name + '/' + basename + '_from_t1_field.nii.gz'
    ase_to_t1_field = dir_name + '/' + basename + '_to_t1_field.nii.gz'
    ase_gm = dir_name + '/' + basename + '_gm.nii.gz'
    shift_im = dir_name + '/' + basename + 'ave_shift'

    if not path.exists(T1_2mm_mask):
        cmd = 'fslmaths ' + T1 + ' -subsamp2 ' + T1_2mm
        system(cmd)

        # Create a heavily eroded mask for registering the ASE QBold data
        mask_cmd = 'fslmaths ' + T1_2mm + ' -bin -kernel box 25 -ero ' + T1_2mm_mask
        system(mask_cmd)

        # Create an inverted mask for FSL anat - don't erode it as it disrupts the registration to std space
        mask_cmd = 'fslmaths ' + T1_2mm + ' -bin -sub 1 -mul -1 ' + T1_2mm_invmask
        system(mask_cmd)

    if not path.exists(anat_dir + 'T1_to_MNI_nonlin_field.nii.gz'):
        # Note that fsl_anat needs to be modified that it still performs registration despite not using bet
        fsl_anat_cmd = 'fsl_anat -i ' + T1_2mm + ' -m ' + T1_2mm_invmask + ' --clobber --nobet'
        system(fsl_anat_cmd)

        # fsl_anat_cmd = 'fsl_anat -d ' + anat_dir + ' -m ' + T1_2mm_invmask
        # system(fsl_anat_cmd)

    if not path.exists(warped_mean_image):
        flirt_cmd = ['flirt', '-in', mean_image, '-ref', T1_2mm, '-dof', '7', '-inweight', brain_mask,
                     '-omat', transform_matrix, '-searchrx', '-20', '20', '-searchry', '-20', '20', '-searchrz', '-20',
                     '20', '-finesearch', '2', '-refweight', T1_2mm_mask]

        subprocess.run(flirt_cmd)

        # Transform from the 2mm space to the ROI version in the anat directory
        cmd = 'convert_xfm -omat ' + transform_2_roi + ' -concat ' + anat_dir + 'T1_orig2roi.mat ' + transform_matrix
        system(cmd)

        # Create the average EPI unwarping map (created separately in SPM)

        warp_type_indicator = 'B'
        if 'hyperv_ase' in image_filename:
            warp_type_indicator = 'H'
        field_fnames = glob(dir_name + '/VDM/*'+warp_type_indicator+'*.nii')

        # Average the two warp maps
        cmd = 'fslmaths ' + field_fnames[0] + ' -add ' + field_fnames[1] + ' -mul 0.5 ' + shift_im
        system(cmd)

        create_t1_warp_cmd = 'convertwarp -r ' + anat_dir + 'T1_to_MNI_nonlin.nii.gz' + ' -o ' + warp_to_std + ' -w ' \
                             + anat_dir + 'T1_to_MNI_nonlin_field.nii.gz' + ' -m ' + transform_2_roi + ' -s ' + shift_im
        system(create_t1_warp_cmd)

        # Evaluate the registration by non-linearly warping the mean qbold image to MNI space
        cmd = 'applywarp -i  ' + mean_image + ' -w ' + warp_to_std + ' -o ' + warped_mean_image + ' -r ' + \
              anat_dir + 'T1_to_MNI_nonlin.nii.gz'
        system(cmd)

    if not path.exists(seg_wm_gm_out):
        cmd = 'convert_xfm -omat ' + transform_matrix_inv + ' -inverse ' + transform_matrix
        system(cmd)
        im_basenames = ['c1T1', 'c2T1']
        seg_ims_out = []
        for seg_basename in im_basenames:
            seg_im = dir_name + '/' + seg_basename + '.nii'
            seg_im_out = dir_name + '/' + basename + '_' + seg_basename
            seg_ims_out.append(seg_im_out)

            cmd = 'flirt -in ' + seg_im + ' -ref ' + mean_image + ' -init ' + transform_matrix_inv + ' -applyxfm ' + \
                  ' -out ' + seg_im_out
            system(cmd)

        cmd = 'fslmaths ' + seg_ims_out[0] + ' -add ' + seg_ims_out[1] + ' -thr 0.5 -bin ' + seg_wm_gm_out
        system(cmd)

    if not path.exists(ase_gm):
        cmd = "convertwarp -r " + T1_2mm + " -o " + ase_to_t1_field + " -s " + shift_im + " -m " + transform_matrix
        system(cmd)

        cmd = "invwarp -w " + ase_to_t1_field + " -o " + t1_to_ase_field + " -r " + mean_image
        system(cmd)

        cmd = "applywarp -i " +  dir_name + "/c1T1.nii -r " + mean_image + " -o " + ase_gm + " -w " + t1_to_ase_field
        system(cmd)

        cmd = "fslmaths " + ase_gm + ' -mas ' + brain_mask + ' -thr 0.5 ' + ase_gm
        system(cmd)

    return warp_to_std, ase_gm


def prepare_image(image_filename):
    # Take the mean across time
    # Apply bet with -R -Z
    # Read the image and mask in, concatenate the mask at the end of the timeseries
    #
    import os.path as path
    import subprocess
    dir_name = path.dirname(image_filename)
    basename = path.basename(image_filename).split('.')[0]

    mean_image = dir_name + '/tmean_' + basename + '.nii.gz'
    brain_mask = dir_name + '/mask_' + basename + '_mask.nii.gz'
    mc_images = dir_name + '/mc_' + basename + '.nii.gz'
    ase_gm = dir_name + '/' + basename + '_gm.nii.gz'

    if path.exists(mc_images) == False:
        mc_cmd = ['mcflirt', '-in', image_filename, '-out', mc_images, '-refvol', '2', '-stages', '4', '-sinc_final']
        subprocess.run(mc_cmd)

    if path.exists(mean_image) == False:
        im_mean_cmd = ['fslmaths', mc_images, '-Tmean', mean_image]
        subprocess.run(im_mean_cmd)

    if path.exists(brain_mask) == False:
        mask_cmd = ['bet', mean_image, dir_name + '/mask_' + basename + '.nii.gz', '-R', '-Z', '-m', '-n']
        subprocess.run(mask_cmd)

    img = nib.load(mc_images)
    qform = img.header.get_qform(coded=False)
    img_data = img.get_fdata()

    mask_img = nib.load(brain_mask)

    gm_img = nib.load(ase_gm)

    img_data = np.concatenate([img_data, np.expand_dims(gm_img.get_fdata(), -1),
                               np.expand_dims(mask_img.get_fdata(), -1)], -1)
    return img_data


def prepare_data(directory, orig_filebasename):
    """
    Parse all the data in the directory and saves it as a single .npz file (we don't have *that* much data).
    @param: orig_filebase is the basename of the file that we are looking for
    """

    from os import remove, path
    import tarfile

    filename = orig_filebasename + '.nii*'
    results = glob(directory + '*/' + filename)

    shape = None
    data = []
    tar_file = directory + '/warp_info' + orig_filebasename + '.tar.gz'
    if path.exists(tar_file):
        remove(tar_file)

    tar = tarfile.open(tar_file, 'x:gz')

    seg_masks_merge_cmd = 'fslmerge -t ' + directory + '/' + orig_filebasename + '_gm'

    for idx, im_filename in enumerate(results):
        image_data = prepare_image(im_filename)
        nonlin, gm_im = register_to_t1(im_filename)
        seg_masks_merge_cmd = seg_masks_merge_cmd + " " + gm_im
        tar.add(nonlin, arcname='nonlin' + str(idx) + '.nii.gz')
        if shape is None:
            shape = image_data.shape

        if shape == image_data.shape:
            data.append(image_data)

    tar.close()

    if len(data) > 0:
        np.save(directory + '/' + orig_filebasename, data)
    system(seg_masks_merge_cmd)

prepare_data(data_dir, "baseline_ase")
# estimate_noise_level(data)
