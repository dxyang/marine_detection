import os
import glob
import math
import multiprocessing
from pathlib import Path
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

def calculate_optical_flow(vid_fp: str, total_procs: int, proc_num: int, frames_to_process: List[int], output_dir: str, lock: multiprocessing.Lock=None, save_img_to_disk: bool = False, resize: bool = False, mp_pbar_update_freq: int = 30):
    vc = cv2.VideoCapture(vid_fp)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    img_output_dir = f"{output_dir}/imgs"
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)

    frame_count = 0
    out_count = 0
    last_frame = None

    # assume frame list is sorted
    min_frame = frames_to_process[0]
    max_frame = frames_to_process[-1]

    print(f"proc num: {proc_num}, num frames: {len(frames_to_process)}")

    import time
    time.sleep(5)

    if total_procs == 1:
        f = h5py.File(f"{output_dir}/cv_flow.hdf", "w")
        print(f"{output_dir}/cv_flow.hdf generated!")
        pbar = tqdm(total=num_frames)
    else:
        f = h5py.File(f"{output_dir}/cv_flow_{proc_num}_{total_procs}.hdf", "w")
        print(f"hdf file for {proc_num} of {total_procs} generated!")
        with lock:
            pbar = tqdm(total=len(frames_to_process), desc=f"[Proc {proc_num}]", position=proc_num, leave=False)

    time.sleep(5)

    # aggregate into single hdf file for faster i/o
    while True:
        ret, frame = vc.read()
        if frame is None:
            break

        # multiproc - skip frames that this process is not responsible for
        if frame_count < min_frame - 1:
            frame_count += 1
            continue
        if frame_count > max_frame:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize:
            gray_frame = cv2.resize(gray_frame, (width // 2, height // 2))

        if last_frame is None:
            last_frame = gray_frame
            if frame_count != 0:
                frame_count += 1
                continue

        # check if we've already processed this frame
        # frame_str = str(frame_count).zfill(6)
        # if frame_str in f:
        #     last_frame = gray_frame
        #     frame_count += 1
        #     out_count += 1
        #     if total_procs == 1:
        #         pbar.update()
        #     else:
        #         if out_count % mp_pbar_update_freq == 0:
        #             with lock:
        #                 pbar.update(mp_pbar_update_freq)
        #     continue

        frame_str = str(frame_count).zfill(6)
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

        if save_img_to_disk:
            torch_flow = torch.from_numpy(chw_flow)
            flow_img = flow_to_image(torch_flow)
            write_png(flow_img, f"{img_output_dir}/flow_{frame_str}.png")

        grp["flow"][:] = chw_flow
        out_count += 1

        last_frame = gray_frame
        frame_count += 1
        if total_procs == 1:
            pbar.update()
        else:
            if out_count % mp_pbar_update_freq == 0:
                with lock:
                    pbar.update(mp_pbar_update_freq)
                f.flush()

    with lock:
        pbar.update(out_count % mp_pbar_update_freq)

    if total_procs != 1:
        print(f"Process {proc_num}/{total_procs} finished")

class ArgumentParser(Tap):
    video: str
    flow_output_dir: str

    num_proc: int = 1

def test(proc_num, lock):
    frames = 100000000
    update_rate = 100
    import time
    with lock:
        pbar = tqdm(total=frames, desc=f"[Proc {proc_num}]", position=proc_num, leave=False)

    for i in range(frames):
        time.sleep(0.5)
        if i % update_rate:
            pbar.update(update_rate)

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

    num_frames += 10

    chunk_size = math.ceil(num_frames / float(args.num_proc))
    all_frames = [i for i in range(num_frames)]
    frame_chunks = []
    for i in range(args.num_proc):
        frame_chunks.append(all_frames[i * chunk_size: min((i + 1) * chunk_size, num_frames)])

    if args.num_proc == 1:
        # single proc
        calculate_optical_flow(args.video, total_procs=1, proc_num=0, frames_to_process=all_frames, output_dir=args.flow_output_dir)
    else:
        lock = multiprocessing.Manager().Lock()
        with multiprocessing.Pool(processes=args.num_proc) as pool:
            mp_args = [(args.video, args.num_proc, i, frame_chunks[i], args.flow_output_dir, lock) for i in range(args.num_proc)]
            pool.starmap(calculate_optical_flow, mp_args)
            # mp_args = [(i, lock) for i in range(args.num_proc)]
            # pool.starmap(test, mp_args)

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

