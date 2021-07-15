# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Code for preparing real data for further analysis
import nibabel as nib
import numpy as np

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
    img_data = img.get_fdata()

    mask_img = nib.load(brain_mask)

    img_data = np.concatenate([img_data, np.expand_dims(mask_img.get_fdata(), -1)], -1)
    return img_data


def prepare_data(directory, orig_filebasename):
    """
    Parse all the data in the directory and saves it as a single .npz file (we don't have *that* much data).
    @param: orig_filebase is the basename of the file that we are looking for
    """
    from glob import glob

    filename = orig_filebasename + '.nii*'
    results = glob(directory + '*/' + filename)

    shape = None
    data = []
    for im_filename in results:
        image_data = prepare_image(im_filename)
        if shape is None:
            shape = image_data.shape

        if shape == image_data.shape:
            data.append(image_data)

    if len(data) > 0:
        np.save(directory + '/' + orig_filebasename, data)


prepare_data(data_dir, "hyperv_ase")
# estimate_noise_level(data)
