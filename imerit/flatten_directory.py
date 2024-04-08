"""
Silly script to flatten a directory, note that it preserves parent directory names in the output filename, replacing slashes with underscores
"""

import glob
import sys
from pathlib import Path
import shutil
from tqdm import tqdm
import os

root_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
filetype = sys.argv[3]

filepaths = glob.iglob(str(root_dir / ("**/*." + filetype)), recursive=True)

root_path_len = len(root_dir.parts)

os.makedirs(output_dir, exist_ok=True)

for filepath in tqdm(filepaths):
    filename = Path(filepath).name

    print(filepath)
    
    extended_filename = "_".join(Path(filepath).parts[root_path_len:]) + "_" + filename
    print(extended_filename)
    output_path = output_dir / extended_filename
    shutil.copyfile(filepath, output_path)

print("done")
