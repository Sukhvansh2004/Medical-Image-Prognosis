import os

# Required filenames in each folder
required_files = [
    "affine_transf.mat",
    "mask.nii.gz",
    "normalized.nii.gz",
    "segm.nii.gz"
]

# Read input and output paths
with open("input_files.txt", "r") as f:
    input_paths = f.read().splitlines()

with open("output_paths.txt", "r") as f:
    output_paths = f.read().splitlines()

# Sanity check
assert len(input_paths) == len(output_paths), "Mismatch in number of lines between input and output files."

# Prepare new lists for keeping entries
filtered_input_paths = []
filtered_output_paths = []

# Check each output path for required files
for input_path, output_path in zip(input_paths, output_paths):
    if not os.path.isdir(output_path):
        # Folder doesn't exist, keep the entry
        filtered_input_paths.append(input_path)
        filtered_output_paths.append(output_path)
        continue

    # Check if all required files are present
    missing = False
    for filename in required_files:
        if not os.path.isfile(os.path.join(output_path, filename)):
            missing = True
            break

    if missing:
        filtered_input_paths.append(input_path)
        filtered_output_paths.append(output_path)

# Write the filtered paths back to the files
with open("input_files_2.txt", "w") as f:
    for path in filtered_input_paths:
        f.write(path + "\n")

with open("output_paths_2.txt", "w") as f:
    for path in filtered_output_paths:
        f.write(path + "\n")

print(f"Filtered out {len(input_paths) - len(filtered_input_paths)} completed entries.")
