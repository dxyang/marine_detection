import os

import datetime
import glob
import math
from pathlib import Path
import pickle
import time
from typing import List

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from tap import Tap
import torch
from torchvision.io import write_png
from torchvision.utils import flow_to_image

from tqdm import tqdm

'''
I screwed up some indexing code when I was generating the multiprocess files
Basically an off by one error.
So file 0 was fine, but file 1 needs to be shifted over by 1, file 2 needs to be
shifted over by 2, etc.
This script is my attempt at fixing it. Sorry. :(

You probably don't ever want to run this script but just leaving here in case it's useful :)
'''

class ArgumentParser(Tap):
    video: str
    flow_dir: str

    region_xywh: Optional[List[int]]

if __name__ == '__main__':
    args = ArgumentParser().parse_args()

    vc = cv2.VideoCapture(args.video)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vc.get(cv2.CAP_PROP_FPS)

    flow_hdf_files = glob.glob(f"{args.flow_dir}/*.hdf")
    num_hdf_files = len(flow_hdf_files)
    assert(num_hdf_files == 12) # for now hardcoded

    for idx in range(num_hdf_files):

        f = h5py.File(f"{args.flow_dir}/cv_flow_{idx}_{num_hdf_files}.hdf", "r+")
        keys = list(f.keys())

        print(f"-----File: cv_flow_{idx}_{num_hdf_files}.hdf")
        print(f"Original: Start: {keys[0]}, End: {keys[-1]}")

        if idx > 1:
            del_keys = keys[:idx - 1]
        keys.reverse()

        if idx == 0 or idx == 1:
            continue

        for k_idx, key in tqdm(enumerate(keys)):
            frame_num = int(key)
            actual_frame_str = str(frame_num + idx - 1).zfill(6)

            if actual_frame_str not in f:
                print(f"Creating group: {actual_frame_str}")
                grp = f.create_group(actual_frame_str)
                grp.create_dataset("flow", shape=(2, height, width), dtype=np.float32)

            if key in del_keys:
                print(f"Deleting group: {key}")
                del f[key]
                continue

            chw_flow = f[key]["flow"][:]
            grp["flow"][:] = chw_flow

        final_keys = list(f.keys())
        print(f"After: Start: {final_keys[0]}, End: {final_keys[-1]}")
