from tap import Tap
import os

import sys
import cv2
from os.path import join, isdir
from pathlib import Path
import subprocess

from tqdm import tqdm
import pandas as pd
import time

"""
Naive batch ffmpeg processor, only cuts videos
"""

class ArgumentParser(Tap):
    input_root_dir: str # root directory of videos
    output_root_dir: str # output root directory for videos
    input_file: str # csv specifying times
    duration: str = "00:00:30" # HH:MM:SS
    ext: str = ".MOV" # video extension type to look for
    flatten_output_dir: bool = False # if true, all outputs in non-nested root directory
    skip_existing: bool = True # skip already generated videos

def parse_time_csv(csv_path: str):
    df = pd.read_csv(args.input_file)
    df_times = df[["FileLocation","Starts.at"]]
    df_times = df_times.rename(columns={"FileLocation":"filepath","Starts.at":"start_time"})
    return df_times
    
def clip_video(input_path: str,
                  output_path: str,
                  start_time: str,
                  duration: str):
    commands = ["ffmpeg", "-ss", start_time, "-i", input_path, "-t", duration, "-loglevel", "error", output_path]
    return subprocess.run(commands).returncode
    
def clip_videos(input_dir: str,
                   output_dir: str,
                   files_and_times: pd.DataFrame,
                   duration: str):

    for ind, row in tqdm(files_and_times.iterrows()):
        input_path = Path(join(input_dir, row["filepath"]))

        start_time = row["start_time"]
        out_name = Path(row["filepath"])

        # include time info in the output name
        t = time.strptime(start_time, "%H:%M:%S")
        out_name = out_name.stem + f"_{t.tm_min}m_{t.tm_sec}s" + out_name.suffix
        output_path = Path(join(output_dir, out_name))
        
        if not isdir(output_path.parent):
            os.makedirs(output_path.parent, exist_ok=True)

        if os.path.exists(output_path):
            continue


        
        clip_video(input_path, output_path, start_time, duration)

        
def time_str_transform(t: str, in_format="%M:%S", out_format="%H:%M:%S"):
    return time.strftime(out_format, time.strptime(t, in_format))
    

if __name__ == "__main__":
    args = ArgumentParser().parse_args()

    df_times = parse_time_csv(args.input_file)
    df_times["start_time"] = df_times.apply(lambda row: time_str_transform(row["start_time"]), axis=1)
    
    clip_videos(args.input_root_dir, args.output_root_dir, df_times, args.duration)
    
