import os
import subprocess
import sys
from tqdm import tqdm

# --- Configuration ---

# --- Paths to your input/output list files ---
INPUT_FILE_LIST = "./input_files_2.txt"  # Contains WSL paths to input NIfTI files
OUTPUT_DIR_LIST = "./output_paths_2.txt" # Contains Windows paths to output directories

# Assumes it's in the same directory as the script. Change if needed.
TURBOPREP_EXECUTABLE = "/home/sukhvansh/DIP/turboprep/turboprep-docker"

# --- Path to the template file (Needs to be a WSL path) ---
TEMPLATE_FILE = "/home/sukhvansh/DIP/MNI152_T1_1mm_brain.nii.gz" # <<< --- IMPORTANT: SET THIS PATH

OPTIONS = ["--modality", "t2"] 
# --- Log file to save command outputs ---
LOG_FILE = "turboprep_processing_log.txt"

# --- End Configuration ---

def windows_to_wsl_path(win_path):
    """Converts a Windows path (e.g., D:\folder) to WSL path (e.g., /mnt/d/folder)."""
    path = win_path.strip()
    # Handle drive letters
    if ':' in path:
        drive, rest = path.split(':', 1)
        path = f"/mnt/{drive.lower()}{rest}"
    # Replace backslashes with forward slashes
    return path.replace('\\', '/')

def read_paths_from_file(filepath):
    """Reads lines from a file and returns them as a list, stripping whitespace."""
    try:
        with open(filepath, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        return paths
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)

def run_processing():
    """Reads paths, runs turboprep command for each, logs output."""

    # --- Basic Checks ---
    if not os.path.exists(TURBOPREP_EXECUTABLE):
         print(f"Error: Turboprep executable not found at {TURBOPREP_EXECUTABLE}")
         print("Please ensure the path is correct and the file has execute permissions.")
         sys.exit(1)

    if TEMPLATE_FILE == "/path/to/your/template/file.nii.gz":
        print("Error: Please set the TEMPLATE_FILE path in the script configuration.")
        sys.exit(1)
    elif not os.path.exists(TEMPLATE_FILE):
        print(f"Error: Template file not found at {TEMPLATE_FILE}")
        sys.exit(1)

    print("Reading input and output paths...")
    input_files = read_paths_from_file(INPUT_FILE_LIST)
    output_dirs_win = read_paths_from_file(OUTPUT_DIR_LIST)

    if len(input_files) != len(output_dirs_win):
        print(f"Error: Mismatch in number of lines between {INPUT_FILE_LIST} ({len(input_files)}) and {OUTPUT_DIR_LIST} ({len(output_dirs_win)}).")
        sys.exit(1)

    if not input_files:
        print("Error: Input file list is empty.")
        sys.exit(1)

    print(f"Found {len(input_files)} files to process.")
    print(f"Logging output to: {LOG_FILE}")

    # --- Processing Loop ---
    with open(LOG_FILE, 'w') as log_f:
        log_f.write(f"--- Starting processing run at {__import__('datetime').datetime.now()} ---\n")

        # Use tqdm for progress bar
        for input_file_wsl, output_dir_win in tqdm(zip(input_files, output_dirs_win), total=len(input_files), desc="Processing Files"):

            # Convert output path and ensure it exists
            output_dir_wsl = windows_to_wsl_path(output_dir_win)
            try:
                os.makedirs(output_dir_wsl, exist_ok=True)
            except OSError as e:
                error_msg = f"Error creating directory {output_dir_wsl}: {e}"
                print(f"\n{error_msg}")
                log_f.write(f"SKIPPING: {input_file_wsl}\n{error_msg}\n")
                log_f.write("-" * 50 + "\n")
                continue # Skip to the next file

            # Check if input file exists before running command
            if not os.path.exists(input_file_wsl):
                error_msg = f"Error: Input file not found: {input_file_wsl}"
                print(f"\n{error_msg}")
                log_f.write(f"SKIPPING: {input_file_wsl}\n{error_msg}\n")
                log_f.write("-" * 50 + "\n")
                continue # Skip to the next file


            # Construct the command as a list
            command = [
                TURBOPREP_EXECUTABLE,
                input_file_wsl,
                output_dir_wsl,
                TEMPLATE_FILE
            ] + OPTIONS

            log_f.write(f"Processing: {input_file_wsl}\n")
            log_f.write(f"Output Dir: {output_dir_wsl}\n")
            log_f.write(f"Command: {' '.join(command)}\n")
            log_f.flush() # Ensure log is written before running command

            try:
                # Run the command, capture output
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True, # Decode stdout/stderr as text
                    check=False # Don't raise exception on non-zero exit code
                )

                # Log stdout and stderr
                log_f.write("--- STDOUT ---\n")
                log_f.write(result.stdout if result.stdout else "[No stdout]\n")
                log_f.write("--- STDERR ---\n")
                log_f.write(result.stderr if result.stderr else "[No stderr]\n")

                if result.returncode != 0:
                    log_f.write(f"### Command failed with exit code: {result.returncode} ###\n")
                    print(f"\nWarning: Command failed for {os.path.basename(input_file_wsl)} (Code: {result.returncode}). Check log.")
                else:
                     log_f.write("### Command completed successfully ###\n")

            except Exception as e:
                error_msg = f"### Python script error during subprocess execution: {e} ###\n"
                print(f"\n{error_msg}")
                log_f.write(error_msg)

            log_f.write("-" * 50 + "\n")
            log_f.flush() # Ensure entry is fully written

    print("\nProcessing finished.")
    print(f"Check {LOG_FILE} for detailed output.")

if __name__ == "__main__":
    run_processing()