# Install required dependencies:
# pip install numpy nibabel scipy scikit-image

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, gaussian_filter
from skimage import filters

# Multi-Scale Retinex with Color Restoration for grayscale images
def msrcr_gray(img, sigma_list=(15, 80, 250), gain=1.0, offset=0.0):
    """
    Apply Multi-Scale Retinex on a single-channel image.
      - img: 2D numpy array (float)
      - sigma_list: list/tuple of scales for Gaussian blur
      - gain: scaling factor for output
      - offset: constant offset added
    Returns a float32 image of same shape.
    """
    # Avoid log of zero
    img_safe = img.astype(np.float32) + 1.0
    log_img = np.log(img_safe)
    retinex = np.zeros_like(img_safe)
    for sigma in sigma_list:
        blur = gaussian_filter(img_safe, sigma=sigma)
        retinex += log_img - np.log(blur + 1e-6)
    # Average over scales
    retinex /= len(sigma_list)
    # Apply gain and offset
    msr = gain * retinex + offset
    return msr.astype(np.float32)

# Optional: robust white-stripe normalization (Shinohara et al. 2014)
def white_stripe_normalize(volume, lower_pct=70, upper_pct=90, mask=None):
    data = volume
    if mask is not None:
        vals = data[mask > 0]
    else:
        vals = data.ravel()
    lo, hi = np.percentile(vals, [lower_pct, upper_pct])
    stripe_vals = vals[(vals >= lo) & (vals <= hi)]
    mean_ws = stripe_vals.mean()
    std_ws = stripe_vals.std() if stripe_vals.std() > 0 else 1.0
    return (data - mean_ws) / std_ws

# Parameters
input_dir = '1'
output_dir = 'normalized_regs_msrcr'

# Target ratio and shape
target_ratio = np.array([182, 218, 182])
target_shape = target_ratio.copy()

# MSRCR parameters
sigma_list = (15, 80, 250)
gain = 1.0
offset = 0.0

# Sharpening parameters
sharpen_radius = 1       # Unsharp mask radius
sharpen_amount = 1.0     # Unsharp mask amount

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith('.nii.gz'):
        continue

    # Load image and data
    filepath = os.path.join(input_dir, filename)
    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)

    # Brain mask: nonzero voxels
    brain_mask = (data > 0)

    # Compute padding to reach target ratio
    current_shape = np.array(data.shape)
    scales = current_shape / target_ratio
    scale = np.max(scales)
    padded_shape = np.ceil(scale * target_ratio).astype(int)
    total_pad = padded_shape - current_shape
    pad_before = (total_pad // 2).astype(int)
    pad_after = (total_pad - pad_before).astype(int)
    pad_widths = list(zip(pad_before, pad_after))
    padded_data = np.pad(data, pad_widths, mode='constant', constant_values=0)
    padded_mask = np.pad(brain_mask, pad_widths, mode='constant', constant_values=0)

    # Resample to target shape (cubic interpolation for intensity)
    zoom_factors = target_shape / padded_data.shape
    resampled_data = zoom(padded_data, zoom_factors, order=3)
    resampled_mask = zoom(padded_mask.astype(np.float32), zoom_factors, order=0) > 0.5

    # Apply Multi-Scale Retinex (MSRCR) slice-by-slice
    msr_data = np.zeros_like(resampled_data)
    for z in range(resampled_data.shape[2]):
        slice_img = resampled_data[:, :, z]
        msr_slice = msrcr_gray(slice_img, sigma_list=sigma_list, gain=gain, offset=offset)
        msr_data[:, :, z] = msr_slice

    # Apply unsharp masking (sharpening) slice-by-slice
    sharpened_data = np.zeros_like(msr_data)
    for z in range(msr_data.shape[2]):
        slice_img = msr_data[:, :, z]
        sharp_slice = filters.unsharp_mask(
            slice_img,
            radius=sharpen_radius,
            amount=sharpen_amount,
            preserve_range=True
        )
        sharpened_data[:, :, z] = sharp_slice

    # Normalize with WhiteStripe on sharpened output
    data_norm = white_stripe_normalize(sharpened_data, lower_pct=70, upper_pct=90, mask=resampled_mask)

    # Save output
    out_img = nib.Nifti1Image(
        data_norm.astype(np.float32), affine=img.affine, header=img.header)
    nib.save(out_img, os.path.join(output_dir, filename))

    print(f"Processed {filename}: padded & resampled, MSRCR + sharpening, then WhiteStripe normalization.")

print("All files processed with cubic resampling, MSRCR, unsharp masking, and WhiteStripe normalization.")
