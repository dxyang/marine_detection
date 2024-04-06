"""
Finds all leaf files that match an extension and copies them to a new output directory with the same structure

Usage:
python copy_filetypes.py <input_dir> <output_dir> <filetype>

Example:
python copy_filetypes.py aws-fish yolo-fish png
"""

import glob
import sys
from pathlib import Path
from os import makedirs
from os.path import isdir
import os

import shutil

root_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
filetype = sys.argv[3]

filepaths = glob.iglob(str(root_dir / f"**/*.{filetype}"), recursive=True)

for filepath in filepaths:
    print(filepath)
    # Remove the root portion of path and replace with output_dir
    root_path_portion = len(root_dir.parts)
    output_path = Path(output_dir, *Path(filepath).parts[root_path_portion:])
    os.makedirs(str(output_path.parent), exist_ok=True)
    shutil.copyfile(filepath, output_path)
