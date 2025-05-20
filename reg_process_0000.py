#!/usr/bin/env python3
import os
import shutil
import argparse

def copy_reg0000_files(src_root, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_root):
        for fname in files:
            if fname.endswith('_reg_0000.nii.gz'):
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(dst_dir, fname)
                print(f"Copying {src_path} → {dst_path}")
                shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively copy all *_reg_0000.nii.gz files to one folder."
    )
    parser.add_argument("--source", help="Root directory to search")
    parser.add_argument("--dest",   help="Destination directory to copy files into")
    args = parser.parse_args()

    copy_reg0000_files(args.source, args.dest)
