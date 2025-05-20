import os

# --- Configuration ---
base_input_dir = r"/home/sukhvansh/DIP/images_registered"
base_output_dir = r"/home/sukhvansh/DIP/images_registered_proc_T2"
input_list_filename = "input_files.txt"
output_list_filename = "output_paths.txt"
# --- End Configuration ---

# Ensure the base output directory exists
os.makedirs(base_output_dir, exist_ok=True)

input_file_paths = []
output_dir_paths = []

print(f"Scanning directory: {base_input_dir}")

# List all items (files and folders) in the base input directory
try:
    items_in_input_dir = os.listdir(base_input_dir)
except FileNotFoundError:
    print(f"Error: Input directory not found: {base_input_dir}")
    exit()
except Exception as e:
    print(f"An error occurred while listing directory contents: {e}")
    exit()

# Iterate through the items found
for item_name in items_in_input_dir:
    item_path = os.path.join(base_input_dir, item_name)

    # Check if the item is a directory (patient folder)
    if os.path.isdir(item_path):
        folder_suffix = item_name # e.g., "Patient-001_week-000-1_reg"

        # Construct the full path for the specific input file (_0001.nii.gz)
        # Assumes the file is INSIDE the patient folder
        input_filename = f"{folder_suffix}_0000.nii.gz"
        full_input_path = os.path.join(item_path, input_filename)

        # Construct the full path for the corresponding output directory
        full_output_path = os.path.join(base_output_dir, folder_suffix)+os.sep

        # Add the paths to our lists
        input_file_paths.append(full_input_path)
        output_dir_paths.append(full_output_path)
        print(f"  Found folder: {folder_suffix}")
        print(f"    -> Input file: {full_input_path}")
        print(f"    -> Output path: {full_output_path}")

# Write the input file paths to the text file
try:
    with open(input_list_filename, 'w') as f_in:
        for path in input_file_paths:
            f_in.write(path + '\n')
    print(f"\nSuccessfully wrote {len(input_file_paths)} input paths to {input_list_filename}")
except Exception as e:
    print(f"Error writing to {input_list_filename}: {e}")


# Write the output directory paths to the text file
try:
    with open(output_list_filename, 'w') as f_out:
        for path in output_dir_paths:
            f_out.write(path + '\n')
    print(f"Successfully wrote {len(output_dir_paths)} output paths to {output_list_filename}")
except Exception as e:
    print(f"Error writing to {output_list_filename}: {e}")

print("\nScript finished.")