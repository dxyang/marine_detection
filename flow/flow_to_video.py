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

class ArgumentParser(Tap):
    video: str
    flow_dir: str

    region_xywh: List[int]

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


    vw = cv2.VideoWriter(f'{args.flow_dir}/optical_flow.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, (width, height))

    for idx in range(num_hdf_files):
        f = h5py.File(f"{args.flow_dir}/cv_flow_{idx}_{num_hdf_files}.hdf", "r")
        for key in tqdm(f.keys()):
            frame_num = int(key)
            # frame_str = str(frame_idx).zfill(6)
            chw_flow = f[key]["flow"][:]

            torch_flow = torch.from_numpy(chw_flow)
            flow_img = flow_to_image(torch_flow)

            hwc_img_np = flow_img.numpy().transpose((1,2,0))
            hwc_bgr_img_np = cv2.cvtColor(hwc_img_np, cv2.COLOR_RGB2BGR)

            vw.write(hwc_bgr_img_np)

    vw.release()

