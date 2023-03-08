from typing import Collection, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def get_3d_go_figure(
    size: Tuple[int, int] = (800, 800),
    x_range: Tuple[int, int] = (-1, 1),
    y_range: Tuple[int, int] = (-1, 1),
    z_range: Tuple[int, int] = (-1, 1),
    show_legend: bool = False,
    aspectmode: str = "cube",
    aspectratio: Dict = None,
):
    # Construct a figure for the scene.
    figure = go.Figure()

    # Determine some parts of layout.
    figure.update_layout(
        autosize=False,
        width=size[0],
        height=size[1],
        showlegend=show_legend,
        margin=dict(l=0, r=0, t=0, b=0),
        scene_aspectmode=aspectmode,
        scene_aspectratio=aspectratio,
        scene=dict(
            xaxis=dict(nticks=4, range=x_range),
            yaxis=dict(nticks=4, range=y_range),
            zaxis=dict(nticks=4, range=z_range),
        ),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1),
    )

    return figure

def plot_trajectory_3d(
    fig: go.Figure,
    points: np.array, # 3 x N
    colorscale: str = 'Viridis',
    color_val: Union[List[float], np.ndarray] = None,
    size: int = 6,
    hovertext: Optional[List[str]] = None,
    name: Optional[str] = None,
    connect_points: bool = False,
    c_range: Tuple[float, float] = None,
    invisible_axes: bool = False,
) -> None:
    xs, ys, zs = points[0], points[1], points[2]
    if color_val is None:
        color_val = np.linspace(0, 1, num=points.shape[1])
    assert len(color_val) == len(xs)

    marker_params = {
        "size": size,
        "color": color_val,
        "colorbar": dict(thickness=10),
        "colorscale": colorscale,
        "opacity": 0.8,
    }

    if c_range is not None:
        marker_params["cmin"] = c_range[0]
        marker_params["cmax"] = c_range[1]

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            marker=marker_params,
            line=dict(
                color='darkblue',
                width=2
            ),
            name=name,
            hovertext=hovertext,
            mode="lines+markers" if connect_points else "markers"
        )
    )

