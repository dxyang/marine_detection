import os
import glob
from pathlib import Path
import pickle
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
    magnitude_threshold: float = 2.0
    framerate_fps: int = 30

if __name__ == '__main__':
    args = ArgumentParser().parse_args()

    vc = cv2.VideoCapture(args.video)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    flow_hdf_files = glob.glob(f"{args.flow_dir}/*.hdf")
    num_hdf_files = len(flow_hdf_files)
    assert(num_hdf_files == 12) # for now hardcoded

    print(f"opening h5py files")
    fs= [h5py.File(f"{args.flow_dir}/cv_flow_{idx}_{num_hdf_files}.hdf", "r") for idx in range(num_hdf_files)]
    print(f"opened h5py files")
    num_flow_frames = 0
    for f in fs:
        print(f"hello!")

    for f in fs:
        print(f)
        num_flow_frames += len(f.keys())

    assert num_flow_frames == num_frames

    import pdb; pdb.set_trace()
    exit()

    roi_mask = np.zeros((height, width))
    rx, ry, rw, rh = args.region_xywh
    roi_mask[ry: ry + rh, rx: rx + rw] = 1.0

    avg_xs = []
    avg_ys = []

    for frame_idx in tqdm(range(num_frames)):
        frame_str = str(frame_idx).zfill(6)
        flow = f["flow"][frame_idx]

        U = flow[0]
        V = flow[1]

        magnitudes = np.sqrt(U * U + V * V)
        mag_idxs_x, mag_idxs_y = np.where(magnitudes > args.magnitude_threshold)
        magnitude_mask = np.zeros_like(U)
        magnitude_mask[mag_idxs_x, mag_idxs_y] = 1.0
        final_mask = np.logical_and(roi_mask, magnitude_mask)

        vecs_x = U[final_mask]
        vecs_y = V[final_mask]

        average_x = np.mean(vecs_x)
        average_y = np.mean(vecs_y)

        avg_xs.append(average_x)
        avg_ys.append(average_y)

    metadata = {
        "avg_xs": avg_xs,
        "avg_ys": avg_ys,
        "magnitude_threshold": args.magnitude_threshold,
        "roi_xywh": args.region_xywh,
        "video": args.video,
    }

    output_dir = Path(flow_np_dir).parent
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    plt.figure()
    plt.plot([t/30.0 for t in range(len(avg_xs))], avg_xs)
    plt.xlabel("Time (s)")
    plt.ylabel("Average U Vector (pixels)")
    plt.savefig(f"{output_dir}/flow_avg_x.png")

    plt.clf(); plt.cla()
    plt.plot([t/30.0 for t in range(len(avg_ys))], avg_ys)
    plt.xlabel("Time (s)")
    plt.ylabel("Average V Vector (pixels)")
    plt.savefig(f"{output_dir}/flow_avg_y.png")
