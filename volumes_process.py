import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Configure logging
type_logger = logging.getLogger('volume_processor')
type_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
type_logger.addHandler(handler)

# Folder containing .nii.gz files
data_dir = 'images_registered'

# Find all brain scans and mask files
brain_files = []
mask_files = []
for root, dirs, files in os.walk(data_dir):
    for fname in files:
        if fname.lower().endswith(('.nii', '.nii.gz')):
            full_path = os.path.join(root, fname)
            if 'mask' in fname.lower():
                mask_files.append(full_path)
            else:
                brain_files.append(full_path)

# Compute volumes (in cubic millimeters) for brains and masks
def compute_volume(nifti_path):
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()  # may raise EOFError on corrupted files
        vox_dims = img.header.get_zooms()[:3]
        voxel_vol = np.prod(vox_dims)
        count = np.sum(data > 0)
        return count * voxel_vol
    except Exception as e:
        type_logger.warning(f"Skipping file due to error: {nifti_path} -> {e}")
        return None

# Collect volumes with progress bars, skipping failures
brain_volumes = {}
for f in tqdm(brain_files, desc="Processing brain files", unit="file"):
    vol = compute_volume(f)
    if vol is not None:
        brain_volumes[f] = vol

mask_volumes = {}
for f in tqdm(mask_files, desc="Processing mask files", unit="file"):
    vol = compute_volume(f)
    if vol is not None:
        mask_volumes[f] = vol

if not brain_volumes:
    raise RuntimeError("No valid brain volumes found. Check your input files.")

# Find largest and smallest brains and their masks
largest_brain = max(brain_volumes, key=brain_volumes.get)
smallest_brain = min(brain_volumes, key=brain_volumes.get)

# Match masks by filename heuristic
def find_matching_mask(brain_fname):
    base = os.path.splitext(os.path.basename(brain_fname))[0]
    # strip additional .nii if present
    base = base[:-4] if base.endswith('.nii') else base
    for m in mask_volumes:
        if base in os.path.basename(m):
            return m
    return None

largest_mask = find_matching_mask(largest_brain)
smallest_mask = find_matching_mask(smallest_brain)

# Report results
print(f"Largest brain: {largest_brain}\n  Volume: {brain_volumes[largest_brain]:.2f} mm^3")
if largest_mask:
    print(f"Corresponding mask: {largest_mask}\n  Volume: {mask_volumes[largest_mask]:.2f} mm^3")
else:
    print("No matching mask found for the largest brain.")

print(f"Smallest brain: {smallest_brain}\n  Volume: {brain_volumes[smallest_brain]:.2f} mm^3")
if smallest_mask:
    print(f"Corresponding mask: {smallest_mask}\n  Volume: {mask_volumes[smallest_mask]:.2f} mm^3")
else:
    print("No matching mask found for the smallest brain.")

# Plot histograms of volumes
plt.figure(figsize=(10, 5))
plt.hist(list(brain_volumes.values()), bins=20, alpha=0.7, label='Brain volumes')
plt.hist(list(mask_volumes.values()), bins=20, alpha=0.7, label='Mask volumes')
plt.xlabel('Volume (mm^3)')
plt.ylabel('Frequency')
plt.title('Histogram of Brain and Mask Volumes')
plt.legend()
plt.tight_layout()
plt.show()
