import os
import glob
import multiprocessing
from pathlib import Path

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

def calculate_optical_flow(vid_fp: str, frame_freq: int, offset: int, output_dir: str, lock: multiprocessing.Lock=None, save_img_to_disk: bool = False, resize: bool = False, mp_pbar_update_freq: int = 30):
    vc = cv2.VideoCapture(vid_fp)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    img_output_dir = f"{output_dir}/imgs"
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)

    frame_count = 0
    out_count = 0
    last_frame = None

    if frame_freq == 1:
        f = h5py.File(f"{output_dir}/cv_flow.hdf", "w")
        pbar = tqdm(total=num_frames)
    else:
        f = h5py.File(f"{output_dir}/cv_flow_{offset}_{frame_freq}.hdf", "a")
        with lock:
            pbar = tqdm(total=num_frames // frame_freq, desc=f"Position {offset}", position=offset, leave=False)

    # aggregate into single hdf file for faster i/o
    while True:
        ret, frame = vc.read()
        if frame is None:
            break
        width = frame.shape[1]
        height = frame.shape[0]

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize:
            gray_frame = cv2.resize(gray_frame, (width // 2, height // 2))

        if last_frame is None:
            last_frame = gray_frame

        # multiproc - skip frames that this process is not responsible for
        if (frame_count - offset) % frame_freq != 0:
            last_frame = gray_frame
            frame_count += 1
            if frame_freq == 1:
                pbar.update()
            continue

        frame_str = str(frame_count).zfill(6)

        # check if we've already processed this frame
        if frame_str in f:
            last_frame = gray_frame
            frame_count += 1
            out_count += 1
            if frame_freq == 1:
                pbar.update()
            else:
                if out_count % mp_pbar_update_freq == 0:
                    with lock:
                        pbar.update(mp_pbar_update_freq)
            continue

        grp = f.create_group(frame_str)
        grp.create_dataset("flow", shape=(2, height, width), dtype=np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            prev=last_frame,
            next=gray_frame,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        chw_flow = np.transpose(flow, (2, 0, 1))

        frame_str = str(frame_count).zfill(6)

        if save_img_to_disk:
            torch_flow = torch.from_numpy(chw_flow)
            flow_img = flow_to_image(torch_flow)
            write_png(flow_img, f"{img_output_dir}/flow_{frame_str}.png")
        grp["flow"][:] = chw_flow
        out_count += 1

        last_frame = gray_frame
        frame_count += 1
        if frame_freq == 1:
            pbar.update()
        else:
            if out_count % mp_pbar_update_freq == 0:
                with lock:
                    pbar.update(mp_pbar_update_freq)
                f.flush()

    with lock:
        pbar.update(out_count % mp_pbar_update_freq)

    if frame_freq != 1:
        print(f"Process {offset}/{frame_freq} finished")

class ArgumentParser(Tap):
    video: str
    flow_output_dir: str

    num_proc: int = 1

if __name__ == '__main__':
    args = ArgumentParser().parse_args()

    flow_output_dir = args.flow_output_dir
    if not os.path.exists(flow_output_dir):
        os.makedirs(flow_output_dir)
    # if not os.path.exists(np_output_dir):
    #     os.makedirs(np_output_dir)

    vc = cv2.VideoCapture(args.video)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # import time
    # start = time.time()
    # pbar = tqdm(total=num_frames)
    # while True:
    #     ret, frame = vc.read()
    #     if frame is None:
    #         break
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     pbar.update()
    # end = time.time()
    # print(end - start)
    # vc = cv2.VideoCapture(args.video)
    # start = time.time()
    # while True:
    #     ret, frame = vc.read()
    #     if frame is None:
    #         break
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # end = time.time()
    # print(end - start)
    # exit()

    if args.num_proc == 1:
        # single proc
        calculate_optical_flow(args.video, frame_freq=1, offset=0, output_dir=args.flow_output_dir)
    else:
        lock = multiprocessing.Manager().Lock()
        with multiprocessing.Pool(processes=args.num_proc) as pool:
            mp_args = [(args.video, args.num_proc, i, args.flow_output_dir, lock) for i in range(args.num_proc)]
            pool.starmap(calculate_optical_flow, mp_args)

    print(f"-------Done with processing flow.")
    # print(f"-------Aggregating into single hdf file.")

    # aggregate into single hdf file for faster i/o
    # np_output_dir = f"{flow_output_dir}/numpy"
    # f = h5py.File(f"{flow_output_dir}/cv_flow.hdf", "w")

    # for idx in tqdm(range(num_frames)):
    #     frame_str = str(idx).zfill(6)
    #     flow_np_path = f"{np_output_dir}/{frame_str}.npy"
    #     # assert os.path.exists(flow_np_path)

    #     flow = np.load(flow_np_path)
    #     if flow.shape != (2, 720, 1280):
    #         import pdb; pdb.set_trace()
    #         print(f"unexpected shape: {frame_str}")
    #         continue

    #     grp = f.create_group(frame_str)
    #     grp.create_dataset("flow", shape=(2, height, width), dtype=np.float32)
    #     grp["flow"][:] = flow

    #     if idx % 1_000 == 0:
    #         f.flush()

