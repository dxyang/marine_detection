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
    metadata: str
    t_start: int = 0
    fps: int = 30
    subsample: int = 20

if __name__ == '__main__':
    args = ArgumentParser().parse_args()

    md = pickle.load(open(args.metadata, 'rb'))
    flow_dir = Path(args.metadata).parent

    num_frames = len(md["avg_xs"])
    num_sec = len(md["avg_xs"]) / 30.0
    t_axis = [t/30.0 for t in range(num_frames)]

    t_ticks = np.arange(0, num_sec, 60)
    t_tick_vals = [str(datetime.timedelta(seconds=s)) for s in t_ticks]



    plt.figure(figsize=(24, 12))
    plt.subplot(211)
    plt.title(Path(md["video"]).name)
    plt.plot(
        t_axis[args.t_start * args.fps:][::args.subsample],
        md["avg_xs"][args.t_start * args.fps:][::args.subsample]
    )
    plt.ylabel("Average X Vector (pixels)")
    plt.ylim(-12, 12)
    plt.xlim(0, math.ceil(len(md["avg_xs"]) / 30.0))
    plt.xticks(t_ticks, [])

    plt.subplot(212)
    plt.plot(
        t_axis[args.t_start * args.fps:][::args.subsample],
        md["avg_ys"][args.t_start * args.fps:][::args.subsample]
    )
    plt.ylabel("Average Y Vector (pixels)")
    plt.ylim(-8, 8)
    plt.xlabel("Time (hh:mm:ss)")
    plt.xlim(0, math.ceil(len(md["avg_xs"]) / 30.0))
    plt.xticks(t_ticks, t_tick_vals, rotation="vertical")

    plt.tight_layout()
    plt.savefig(f"{flow_dir}/flow_avg_xy.png")
