import os
import collections

# --- Configuration ---
# !! IMPORTANT: Replace this with the actual path to your parent folder !!
# Use raw string (r"...") or forward slashes for Windows paths
parent_directory = r"images_registered_proc"

# List of files expected in each subdirectory
expected_files = [
    "affine_transf.mat",
    "mask.nii.gz",
    "normalized.nii.gz",
    "segm.nii.gz"
]
# --- End Configuration ---

# Initialize dictionaries to store results
missing_files_map = {}  # Stores {folder_path: [list_of_missing_files]}
missing_file_counts = collections.Counter() # Counts how many folders miss each specific file

print(f"Scanning subdirectories inside: {parent_directory}\n")

# Check if the parent directory exists
if not os.path.isdir(parent_directory):
    print(f"Error: Parent directory not found at '{parent_directory}'")
else:
    # Iterate through items in the parent directory
    for item_name in os.listdir(parent_directory):
        item_path = os.path.join(parent_directory, item_name)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            subdirectory_path = item_path
            print(f"Checking: {subdirectory_path}")

            # List files actually present in the subdirectory
            try:
                present_files = set(os.listdir(subdirectory_path))
            except OSError as e:
                print(f"  Warning: Could not read directory {subdirectory_path}. Error: {e}")
                continue # Skip this directory

            # Check for missing files
            currently_missing = []
            for expected_file in expected_files:
                if expected_file not in present_files:
                    currently_missing.append(expected_file)
                    missing_file_counts[expected_file] += 1 # Increment count for this missing file

            # If any files were missing, record it
            if currently_missing:
                missing_files_map[subdirectory_path] = currently_missing
                print(f"  -> Missing: {', '.join(currently_missing)}")
            else:
                print("  -> OK (All expected files found)")


    # --- Print Summary ---
    print("\n--- Scan Complete ---")

    if not missing_files_map:
        print("\nExcellent! All subdirectories contain the expected files.")
    else:
        print("\nFolders with Missing Files:")
        for folder, missing in missing_files_map.items():
            print(f"- {folder}: Missing {', '.join(missing)}")

        print("\nSummary of Missing Files Across All Folders:")
        if not missing_file_counts:
             print("No files missing from any folder.")
        else:
            for file_name, count in missing_file_counts.items():
                print(f"- '{file_name}': Missing from {count} folder(s)")

print("\n---------------------")