import nibabel as nib
import numpy as np
from scipy.ndimage import label
import os

def get_largest_connected_component(binary_img):
    labeled_array, num_features = label(binary_img)
    if num_features == 0:
        raise ValueError("No connected components found.")
    
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # background label
    largest_label = sizes.argmax()
    
    largest_component = labeled_array == largest_label
    return largest_component.astype(np.uint8)

def extract_brain_mask(input_path, output_path=None, threshold=0.1):
    # Load the NIfTI image
    img = nib.load(input_path)
    data = img.get_fdata()

    # Normalize data and apply threshold to binarize
    normalized = (data - data.min()) / (data.max() - data.min())
    binary_mask = normalized > threshold

    # Get largest connected component
    brain_mask = get_largest_connected_component(binary_mask)

    # Save the mask as a new NIfTI file
    brain_mask_img = nib.Nifti1Image(brain_mask, img.affine, img.header)
    if not output_path:
        output_path = os.path.splitext(os.path.splitext(input_path)[0])[0] + '_brain_mask.nii.gz'
    nib.save(brain_mask_img, output_path)
    print(f"Largest brain mask saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract largest brain mask from .nii.gz file")
    parser.add_argument("--input_file", help="Path to input .nii.gz file")
    parser.add_argument("--output", help="Path to save output mask file")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for binarization")

    args = parser.parse_args()
    extract_brain_mask(args.input_file, args.output, args.threshold)
