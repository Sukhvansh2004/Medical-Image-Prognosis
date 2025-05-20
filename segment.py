import os
import subprocess

# --- Configuration ---
input_dir = "reg_0000"
output_dir = "reg_0000_process"
# Example command: modify as needed; {input_path} and {output_path} will be substituted
command_template = "mri_synthseg --i \"{input_path}\" --o \"{output_path}\" --fast --threads 8 --resample 1"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file in the input directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Skip if output file already exists
    if os.path.exists(output_path):
        print(f"Skipping {filename} (already processed)")
        continue

    try:
        # Format the command with input and output paths
        command = command_template.format(input_path=input_path, output_path=output_path)
        print(f"Processing {input_path} -> {output_path}")
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully processed {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command for {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error for {filename}: {e}")
