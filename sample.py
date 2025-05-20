import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# Parameters
input_dir = 'reg_0000_process'
output_dir = 'reg_downsample'

# Target ratio and shape
target_ratio = np.array([182, 218, 182])
target_shape = target_ratio.copy()

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each NIfTI file in the input directory
for filename in os.listdir(input_dir):
    if not filename.endswith('.nii.gz'):
        continue

    # Load image
    filepath = os.path.join(input_dir, filename)
    img = nib.load(filepath)
    data = img.get_fdata()

    # Compute current shape
    current_shape = np.array(data.shape)

    # Determine scale to apply to target_ratio so it's >= current_shape
    scales = current_shape / target_ratio
    scale = np.max(scales)

    # Compute padded shape: ceil(scale * target_ratio)
    padded_shape = np.ceil(scale * target_ratio).astype(int)

    # Compute padding widths on each side for each axis
    total_pad = padded_shape - current_shape
    pad_before = (total_pad // 2).astype(int)
    pad_after = (total_pad - pad_before).astype(int)
    pad_widths = list(zip(pad_before, pad_after))

    # Apply zero padding
    padded_data = np.pad(data, pad_widths, mode='constant', constant_values=0)

    # Compute zoom factors to downsample to target_shape
    zoom_factors = target_shape / padded_data.shape

    # Resample using nearest-neighbor interpolation (order=0) to preserve discrete labels
    resampled_data = zoom(padded_data, zoom_factors, order=0)
    # Ensure integer labels
    resampled_data = resampled_data.astype(data.dtype)

    # Save output with same filename
    out_img = nib.Nifti1Image(resampled_data, affine=img.affine, header=img.header)
    nib.save(out_img, os.path.join(output_dir, filename))

    print(f"Processed {filename}: from {current_shape.tolist()} to {target_shape.tolist()}")

print("All files processed with nearest-neighbor interpolation.")
