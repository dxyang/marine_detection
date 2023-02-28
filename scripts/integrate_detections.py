import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tap import Tap
from tqdm import tqdm

class ArgumentParser(Tap):
    data_pkl: str
    detection_dir: str
    confidence_threshold: float


def process_detection_txt_file(file: Path, threshold: float):
    count = 0
    with open(file) as fp:
        for line in fp:
            obj_class, x, y, w, h, conf = line.split()
            if float(conf) >= threshold:
                count += 1
    return count

if __name__ == '__main__':
    args = ArgumentParser().parse_args()

    model_name = Path(args.detection_dir).parent.stem

    df = pd.read_pickle(args.data_pkl)
    num_images = len(df.index)

    '''
    get all the detections
    '''
    all_counts = np.zeros(num_images)
    detection_txt_files = glob.glob(f"{args.detection_dir}/*.txt")
    img_frames = []
    for txt_file in tqdm(detection_txt_files):
        txt_file_path = Path(txt_file)
        img_num = int(txt_file_path.stem)
        fishcount = process_detection_txt_file(txt_file_path, args.confidence_threshold)
        all_counts[img_num] = fishcount

    '''
    add counts to pkl
    '''
    df[str(model_name)] = all_counts
    df.to_pickle(args.data_pkl)

    print(f"updated {args.data_pkl} with fish counts from model {model_name} at threshold {args.confidence_threshold}")
