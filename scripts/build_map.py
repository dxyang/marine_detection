from re import I
from typing import List

import rosbag
import pandas as pd
from tap import Tap
from tqdm import tqdm
import torch

class ArgumentParser(Tap):
    bagfile: str  # path to ros bag file
    # topics: List[str] #

def process_msg(msg, row=None, prefix="", t=None, ref_time=None):
    if row is None:
        row = {}
    if t is not None:
        row["bag_time"] = t
        if ref_time is not None:
            row["offset_time"] = t - ref_time
    if isinstance(msg, int) or isinstance(msg, float) or isinstance(msg, str):
        row[prefix] = msg
    elif isinstance(msg, list) or isinstance(msg, tuple):
        for i, e in enumerate(msg):
            process_msg(e, row=row, prefix=f"{prefix}[{str(i)}]")
    else:
        if len(prefix) > 0 and not prefix.endswith('.'):
            prefix = prefix + '.'
        for k in msg.__slots__:
            process_msg(getattr(msg, k), row=row, prefix=f"{prefix}{k}")
    return row

def process_odom_msg(msg, t=None, ref_time=None):
    '''
    for now, all we care about is the timestamp and pose of the vehicle
    '''
    row = {}
    if t is not None:
        row["bag_time"] = t
        if ref_time is not None:
            row["offset_time"] = t - ref_time

    pos_xyz = msg.pose.pose.position
    orientation_xyzw = msg.pose.pose.orientation

    row["x"] = pos_xyz.x
    row["y"] = pos_xyz.y
    row["z"] = pos_xyz.z
    row["qx"] = orientation_xyzw.x
    row["qy"] = orientation_xyzw.y
    row["qz"] = orientation_xyzw.z
    row["qw"] = orientation_xyzw.w

    return row

def process_img_msg(msg, t=None, ref_time=None):
    '''
    we can process the images through the ultralytics yolo pipeline, but we need to associate images to timestamps
    '''
    row = {}
    if t is not None:
        row["bag_time"] = t
        if ref_time is not None:
            row["offset_time"] = t - ref_time
    return row


def to_csv(bag, topic, out):
    """

    :type bag: rosbag.Bag
    """
    rows = []
    ref_time = bag.get_start_time()
    for topic, msg, t in tqdm(bag.read_messages(topics=[topic]), desc="Extracting messages"):
        rows.append(process_msg(msg, t=t.to_sec(), ref_time=ref_time))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)


if __name__ == '__main__':
    args = ArgumentParser().parse_args()

    bag = rosbag.Bag(args.bagfile, "r")
    topics = bag.get_type_and_topic_info()[1].keys()

    odom_rows = []
    forward_cam_rows = []
    downward_cam_rows = []

    ref_time = bag.get_start_time()
    process_topics = [
        '/warpauv_2/cameras/downward/rgb/image_stream/h264',
        '/warpauv_2/cameras/forward/rgb/image_stream/h264',
        '/warpauv_2/metashape_odom',
    ]
    for topic in process_topics:
        assert topic in topics

    for topic, msg, t in tqdm(bag.read_messages(topics=process_topics), desc="Extracting messages"):
        if "odom" in topic:
            odom_rows.append(process_odom_msg(msg, t=t.to_sec(), ref_time=ref_time))
        elif "downward" in topic:
            downward_cam_rows.append(process_img_msg(msg, t=t.to_sec(), ref_time=ref_time))
        elif "forward" in topic:
            forward_cam_rows.append(process_img_msg(msg, t=t.to_sec(), ref_time=ref_time))

    for idx in range(len(forward_cam_rows)):
        forward_cam_rows[idx]["image_idx"] = idx
    for idx in range(len(downward_cam_rows)):
        downward_cam_rows[idx]["image_idx"] = idx

    odom_df = pd.DataFrame(odom_rows)
    forward_cam_df = pd.DataFrame(forward_cam_rows)
    downward_cam_df = pd.DataFrame(downward_cam_rows)

    merged_df = pd.concat([odom_df, forward_cam_df, downward_cam_df])
    merged_df = merged_df.sort_values(by="offset_time").reset_index(drop=True)

    '''
    interpolate xyz position
    '''


    # with open(args.output_file, 'w') as out:
        # to_csv(bag, args.topic, out)
    bag.close()

