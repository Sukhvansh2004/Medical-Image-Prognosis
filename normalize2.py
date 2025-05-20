# Install required dependencies:
# pip install numpy nibabel scipy scikit-image

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage import exposure, filters

# WhiteStripe normalization
def white_stripe_normalize(volume, lower_pct=70, upper_pct=90, mask=None):
    data = volume
    if mask is not None and mask.any():
        vals = data[mask]
    else:
        vals = data.ravel()
    if vals.size == 0:
        return data
    lo, hi = np.percentile(vals, [lower_pct, upper_pct])
    stripe = vals[(vals >= lo) & (vals <= hi)]
    mean_ws = stripe.mean()
    std_ws = stripe.std() if stripe.std() > 0 else 1.0
    return (data - mean_ws) / std_ws

# Parameters
input_dir = '1'
output_dir = 'normalized_regs2'
target_shape = np.array([182, 218, 182], dtype=float)

# CLAHE parameters
clahe_clip_limit = 0.03
clahe_kernel_size = None  # defaults to image size / 8

# Sharpening parameters
sharpen_radius = 1.0
sharpen_amount = 1.0

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith('.nii.gz'):
        continue

    img = nib.load(os.path.join(input_dir, fname))
    data = img.get_fdata().astype(np.float32)
    mask = data > 0

    # Pad to maintain ratio
    orig_shape = np.array(data.shape, dtype=float)
    scale = np.max(orig_shape / target_shape)
    pad_shape = np.ceil(scale * target_shape).astype(int)
    pad_total = pad_shape - data.shape
    pads = [(pad_total[i]//2, pad_total[i]-pad_total[i]//2) for i in range(3)]
    pd = np.pad(data, pads, mode='constant', constant_values=0)
    pm = np.pad(mask, pads, mode='constant', constant_values=0)

    # Resample (cubic for intensity, nearest for mask)
    factors = target_shape / np.array(pd.shape, dtype=float)
    rd = zoom(pd, factors, order=3)
    rm = zoom(pm.astype(float), factors, order=0) > 0.5

    # Prepare output array
    proc = np.copy(rd)

    # Process slice-by-slice
    for z in range(rd.shape[2]):
        slice_img = rd[:, :, z]
        slice_mask = rm[:, :, z]
        if not slice_mask.any():
            continue
        # Crop ROI
        ys, xs = np.where(slice_mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = slice_img[y0:y1, x0:x1]
        # Normalize crop to [0,1]
        mn, mx = crop.min(), crop.max()
        if mx <= mn:
            continue
        norm_crop = (crop - mn) / (mx - mn)
        # CLAHE
        clahe = exposure.equalize_adapthist(
            norm_crop,
            clip_limit=clahe_clip_limit,
            kernel_size=clahe_kernel_size
        )
        # Sharpen
        sharp = filters.unsharp_mask(
            clahe,
            radius=sharpen_radius,
            amount=sharpen_amount,
            preserve_range=True
        )
        # Map back
        proc_crop = sharp * (mx - mn) + mn
        # Insert back only brain region
        region_mask = slice_mask[y0:y1, x0:x1]
        out = proc[:, :, z]
        out[y0:y1, x0:x1][region_mask] = proc_crop[region_mask]
        proc[:, :, z] = out

    # Normalize using WhiteStripe
    normed = white_stripe_normalize(proc, mask=rm)

    # Save
    out_img = nib.Nifti1Image(normed.astype(np.float32), img.affine, img.header)
    nib.save(out_img, os.path.join(output_dir, fname))
    print(f"{fname} -> processed, shape {proc.shape}")

print("All files processed.")
