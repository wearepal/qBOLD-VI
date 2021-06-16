# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Code for preparing real data for further analysis
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/Users/is321/Documents/Data/qBold/hyperv_data/'

subjects = ['CISC17352', 'CISC17543', 'CISC17987', 'CISC19890']

img = nib.load(data_dir + subjects[3] + '/hyperv_ase.nii.gz')
data = img.get_fdata()


def estimate_noise_level(data):
    # Estimate the SNR from the corner of the image (0 true signal)
    corner = data[1:18, 1:18, :, :]
    # Reshape
    corner = corner.reshape(-1, 11)
    # Normalise by tau =0 image and take the log
    corner = np.log(corner / corner[:, 2:3])
    # Log at the std-dev
    print(np.std(corner[:, 3:]))
    # plot the histogram for different taus
    plt.hist(corner)
    plt.show()

    # Look at the whole image histogram
    whole_image = data.reshape(-1, 11) / data.reshape(-1, 11)[:, 2:3]
    plt.hist(whole_image)
    plt.show()

estimate_noise_level(data)

