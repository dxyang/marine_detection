import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tap import Tap
from tqdm import tqdm

from scripts.utils import get_3d_go_figure, plot_trajectory_3d

class ArgumentParser(Tap):
    data_pkl: str
    model_name: str

    view_trajectory: bool = False

if __name__ == '__main__':
    args = ArgumentParser().parse_args()

    df = pd.read_pickle(args.data_pkl)

    # sort by increasing number of fish counts for plotting
    df = df.sort_values(args.model_name, ascending=True)

    '''
    view trajectory
    '''
    robot_xyz = df[['x', 'y', 'z']] # N x 3
    min_xyz = np.min(robot_xyz, axis=0)
    max_xyz = np.max(robot_xyz, axis=0)

    plot_xyzs = np.array(robot_xyz.T) # 3 x N

    traj_fig = get_3d_go_figure(
        x_range=(math.floor(min_xyz.x), math.ceil(max_xyz.x)),
        y_range=(math.floor(min_xyz.y), math.ceil(max_xyz.y)),
        z_range=(math.floor(min_xyz.z), math.ceil(max_xyz.z)),
    )
    plot_trajectory_3d(traj_fig, plot_xyzs, color_val=df.index.to_numpy())
    traj_fig.write_html("trajectory.html", auto_open=args.view_trajectory)

    '''
    plot fish counts
    '''
    COLORMAP_MAX = 15
    COLORMAP_MIN = 0

    biomap_fig = get_3d_go_figure(
        x_range=(math.floor(min_xyz.x), math.ceil(max_xyz.x)),
        y_range=(math.floor(min_xyz.y), math.ceil(max_xyz.y)),
        z_range=(math.floor(min_xyz.z), math.ceil(max_xyz.z)),
    )

    fish_counts = df[args.model_name].to_numpy()
    plot_trajectory_3d(biomap_fig, plot_xyzs, color_val=fish_counts, c_range=(COLORMAP_MIN, COLORMAP_MAX))
    biomap_fig.write_html(f"biomap_{args.model_name}.html", auto_open=args.view_trajectory)

