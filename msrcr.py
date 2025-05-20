import os
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import zoom

# Optional: robust white-stripe normalization (if desired)
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


def load_nii_image(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine, nii.header


def pad_and_resample(volume, mask, target_shape):
    # Compute padding for target ratio
    current = np.array(volume.shape)
    target = np.array(target_shape)
    scales = current / target
    scale = np.max(scales)
    padded_shape = np.ceil(scale * target).astype(int)

    total_pad = padded_shape - current
    pad_before = (total_pad // 2).astype(int)
    pad_after = (total_pad - pad_before).astype(int)
    pad_widths = list(zip(pad_before, pad_after))

    padded_vol = np.pad(volume, pad_widths, mode='constant', constant_values=0)
    padded_mask = np.pad(mask, pad_widths, mode='constant', constant_values=0)

    # Compute zoom factors to reach target_shape
    zooms = target / padded_vol.shape
    resampled_vol = zoom(padded_vol, zooms, order=3)
    resampled_mask = zoom(padded_mask.astype(np.float32), zooms, order=0) > 0.5

    return resampled_vol, resampled_mask


def get_brain_mask(slice_2d):
    # Otsu-based mask per slice
    norm8 = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(norm8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def apply_clahe(slice_2d, clip_limit=2.0, tile_grid_size=(8, 8)):
    norm = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(norm)


def apply_msrcr(slice_2d, sigma_list=(15, 80, 250), gain=1.0, offset=0):
    img = slice_2d.astype(np.float32) + 1.0
    retinex = sum(
        np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        for sigma in sigma_list
    )
    retinex /= len(sigma_list)
    result = gain * retinex + offset
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def process_slice(slice_2d, method):
    if method == "clahe":
        return apply_clahe(slice_2d)
    elif method == "msrcr":
        return apply_msrcr(slice_2d)
    elif method == "clahe_msrcr":
        clahe_img = apply_clahe(slice_2d)
        msrcr_img = apply_msrcr(slice_2d)
        return 0.6 * clahe_img + 0.4 * msrcr_img
    else:
        raise ValueError(f"Invalid method: {method}")


def process_and_save(input_dir, output_dir, method, target_shape):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
            continue

        path = os.path.join(input_dir, fname)
        vol, affine, header = load_nii_image(path)
        brain_mask_vol = (vol > 0)

        # Pad & resample whole volume
        vol_rs, mask_rs = pad_and_resample(vol, brain_mask_vol, target_shape)

        # Prepare output array
        out_vol = np.zeros_like(vol_rs, dtype=np.uint8)

        # Process slice-by-slice
        for z in range(vol_rs.shape[2]):
            sl = vol_rs[:, :, z]
            m = mask_rs[:, :, z]
            masked_input = np.zeros_like(sl)
            masked_input[m] = sl[m]
            enhanced = process_slice(masked_input, method)
            masked_out = np.zeros_like(enhanced)
            masked_out[m] = enhanced[m]
            out_vol[:, :, z] = masked_out

        # Optional: WhiteStripe normalization
        # out_vol_norm = white_stripe_normalize(out_vol, mask=mask_rs)

        base = os.path.splitext(os.path.basename(fname))[0]
        out_fname = f"{base}_{method}_res{target_shape[0]}x{target_shape[1]}x{target_shape[2]}.nii.gz"
        out_path = os.path.join(output_dir, out_fname)

        nib.save(nib.Nifti1Image(out_vol, affine, header), out_path)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    # Configuration
    input_dir = '1'
    output_dir = 'enhanced_resampled_msrcr_clahe'
    method = 'clahe_msrcr'  # or 'clahe', 'msrcr'
    target_shape = (182, 218, 182)
    process_and_save(input_dir, output_dir, method, target_shape)
