#!/usr/bin/env python3
import os
import subprocess
import sys
from tqdm import tqdm
from datetime import datetime

# --- Configuration ---
INPUT_LIST = "./input_files_2.txt"
OUTPUT_LIST = "./output_paths_2.txt"
TEMPLATE_FILE = "/home/sukhvansh/DIP/MNI152_T1_1mm_brain.nii.gz"
DOCKER_IMAGE = "lemuelpansh/turboprep:latest"
OPTIONS = ["--modality", "t1"]
LOG_FILE = "turboprep_processing_log.txt"

# --- Helpers ---
def windows_to_wsl(path: str) -> str:
    """
    Converts a Windows path (e.g., D:\folder\file.nii) to
    a WSL path (e.g., /mnt/d/folder/file.nii).
    """
    drive, rest = path.split(":", 1)
    # replace backslashes in the remainder before formatting
    unix_rest = rest.replace("\\", "/")
    return f"/mnt/{drive.lower()}{unix_rest}"

def read_lines(filepath: str):
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)

# --- Main Processing ---
def main():
    # Pre-pull image
    subprocess.run(["docker", "pull", DOCKER_IMAGE], check=False)

    inputs   = read_lines(INPUT_LIST)
    outputs  = read_lines(OUTPUT_LIST)
    if len(inputs) != len(outputs):
        print("Error: input/output count mismatch.")
        sys.exit(1)

    with open(LOG_FILE, 'w') as log:
        log.write(f"Start: {datetime.now()}\n")
        gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        log.write("--- GPU STATUS ---\n" + gpu.stdout + "\n")

        for in_wsl, out_win in tqdm(zip(inputs, outputs), total=len(inputs), desc="Processing"):
            out_wsl = windows_to_wsl(out_win)
            os.makedirs(out_wsl, exist_ok=True)

            if not os.path.exists(in_wsl):
                log.write(f"Missing input: {in_wsl}\n" + "-"*40 + "\n")
                continue

            # build docker command
            cmd = [
                "docker", "run", "--rm", "--gpus", "all",
                "-v", f"{os.path.dirname(in_wsl)}:/app/input",
                "-v", f"{out_wsl}:/app/output",
                "-v", f"{os.path.dirname(TEMPLATE_FILE)}:/app/template",
                DOCKER_IMAGE,
                f"/app/input/{os.path.basename(in_wsl)}",
                "/app/output",
                f"/app/template/{os.path.basename(TEMPLATE_FILE)}",
            ] + OPTIONS

            log.write("Running: " + " ".join(cmd) + "\n")
            result = subprocess.run(cmd, capture_output=True, text=True)
            log.write("--- STDOUT ---\n" + (result.stdout or "[No stdout]\n"))
            log.write("--- STDERR ---\n" + (result.stderr or "[No stderr]\n"))
            log.write(f"Exit code: {result.returncode}\n" + "-"*40 + "\n")

    print(f"Done. See {LOG_FILE}")

if __name__ == "__main__":
    main()
