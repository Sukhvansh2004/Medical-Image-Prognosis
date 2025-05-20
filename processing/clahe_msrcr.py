import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def load_nii_image(file_path):
    """
    Load a NIfTI image and return the 3D numpy array and affine.
    """
    nii_img = nib.load(file_path)
    img_data = nii_img.get_fdata()
    return img_data, nii_img.affine


def get_brain_mask(slice_2d):
    """
    Compute a rough brain mask from a 2D slice using Otsu thresholding and morphological closing.
    Returns a boolean mask where True indicates brain region.
    """
    slice_uint8 = cv2.normalize(slice_2d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(slice_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def apply_clahe_to_slice(slice_2d, clip_limit=2.0, tile_grid_size=(8, 8)):
    slice_norm = cv2.normalize(slice_2d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    slice_uint8 = slice_norm.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(slice_uint8)


def apply_msrcr_to_slice(slice_2d, sigma_list=(15, 80, 250), gain=1.0, offset=0):
    img = slice_2d.astype(np.float32) + 1.0  # avoid log(0)
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += (np.log10(img) - np.log10(blur))
    retinex /= len(sigma_list)
    retinex = gain * retinex + offset
    ret_norm = cv2.normalize(retinex, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return ret_norm.astype(np.uint8)


def apply_clahe_msrcr_to_slice(slice_2d,
                                clip_limit=2.0, tile_grid_size=(8, 8),
                                sigma_list=(15, 80, 250), gain=1.0, offset=0):
    clahe_slice = apply_clahe_to_slice(slice_2d, clip_limit, tile_grid_size)
    return apply_msrcr_to_slice(clahe_slice, sigma_list, gain, offset)


def visualize_results(original, clahe, msrcr, clahe_msrcr, slice_index):
    """
    Show original, CLAHE, MSRCR, and CLAHE+MSRCR slices in a 2x2 grid,
    using a unified intensity scale across all images.
    """
    titles = [f"Original Slice {slice_index}", "CLAHE", "MSRCR", "CLAHE + MSRCR"]
    imgs = [original, clahe, msrcr, clahe_msrcr]

    # Compute global intensity min and max
    global_min = min(np.min(img) for img in imgs)
    global_max = max(np.max(img) for img in imgs)

    plt.figure(figsize=(12, 10))
    for i, (img, title) in enumerate(zip(imgs, titles), 1):
        plt.subplot(2, 2, i)
        plt.imshow(img, cmap='gray', vmin=global_min, vmax=global_max)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_histograms(images, titles, mask, slice_index):
    """
    Plot histograms of the given masked image arrays.
    """
    plt.figure(figsize=(12, 10))
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(2, 2, i)
        # Flatten only masked brain pixels
        data = img[mask].ravel()
        plt.hist(data, bins=256, range=(0, 255))
        plt.title(f"{title} Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    nii_path = "/home/sukhvansh/DIP/1/Patient-001_week-044_reg_0000_resampled.nii.gz"
    slice_index = 60  # Change this to pick different slice

    try:
        volume, affine = load_nii_image(nii_path)
        if slice_index >= volume.shape[2]:
            raise ValueError(f"Slice index out of range. Max index: {volume.shape[2]-1}")

        original_slice = volume[:, :, slice_index]
        mask = get_brain_mask(original_slice)

        original_masked = np.zeros_like(original_slice)
        original_masked[mask] = original_slice[mask]

        clahe_slice = apply_clahe_to_slice(original_masked)
        msrcr_slice = apply_msrcr_to_slice(original_masked)
        combined_slice = apply_clahe_msrcr_to_slice(original_masked)

        clahe_masked = np.zeros_like(clahe_slice)
        msrcr_masked = np.zeros_like(msrcr_slice)
        combined_masked = np.zeros_like(combined_slice)
        clahe_masked[mask] = clahe_slice[mask]
        msrcr_masked[mask] = msrcr_slice[mask]
        combined_masked[mask] = combined_slice[mask]

        # Visualize images
        visualize_results(original_masked, clahe_masked, msrcr_masked, combined_masked, slice_index)
        # Plot histograms
        titles = [f"Original {slice_index}", "CLAHE", "MSRCR", "CLAHE+MSRCR"]
        images = [original_masked, clahe_masked, msrcr_masked, combined_masked]
        plot_histograms(images, titles, mask, slice_index)

        # Save outputs
        out_dir = '.'
        base_names = ['original', 'clahe', 'msrcr', 'clahe_msrcr']
        for name, img in zip(base_names, images):
            out_file = os.path.join(out_dir, f"brain_{name}_{slice_index}.png")
            cv2.imwrite(out_file, cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        print(f"Processed slices and histograms saved to {out_dir}")

    except Exception as e:
        print(f"Error: {e}")
