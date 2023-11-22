from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from .dataframe import (
    get_looking_intervals,
    get_fixation_intervals,
    get_response_data,
    get_task_properties,
    get_all_task_failure_intervals,
)
from .interval import merge_intervals


def plot_intervals(
    intervals,
    ax=None,
    alpha=0.25,
    color="red",
    ymin=0,
    ymax=1,
    fill=True,
    linewidth=0.0,
):
    if ax is None:
        ax = plt.gca()
    intervals = np.array(intervals)
    assert len(intervals.shape) == 2  # should be (N,2)
    assert intervals.shape[-1] == 2  # invalid interval data

    for interval in intervals:
        ax.axvspan(
            *interval,
            alpha=alpha,
            color=color,
            linewidth=linewidth,
            ymin=ymin,
            ymax=ymax,
            fill=fill,
        )


def plot_looking(
    data,
    ax=None,
    plot_response=True,
    plot_fixation=True,
    hold_action=False,
    ymin=0.05,
    ymax=0.95,
    xlim_offset=(10, 1),
):
    if ax is None:
        ax = plt.gca()

    start_time, finish_time = (
        data["start_time"] + xlim_offset[0],
        data["finish_time"] + xlim_offset[1],
    )

    ax.add_patch(
        Rectangle(
            (start_time, ymin),
            finish_time - start_time,
            ymax - ymin,
            edgecolor="black",
            fill=False,
            lw=1,
        )
    )

    looking_intervals = get_looking_intervals(data)
    for task in looking_intervals["task"].unique():
        li = looking_intervals[looking_intervals["task"] == task]
        plot_intervals(
            li[["t1", "t2"]].to_numpy(),
            color=get_task_properties(task)["color"],
            ax=ax,
            ymin=ymin,
            ymax=ymax,
        )

    if plot_fixation:
        fixation_intervals = get_fixation_intervals(data)
        for task in fixation_intervals["task"].unique():
            li = fixation_intervals[fixation_intervals["task"] == task]
            plot_intervals(
                li[["t1", "t2"]].to_numpy(),
                color=get_task_properties(task)["color"],
                ymin=ymin,
                ymax=ymax,
                ax=ax,
            )

    if plot_response:
        response_data = get_response_data(
            data["keyboard_data"], data["click_data"], hold_action=hold_action
        )
        response_style = dict(mouse_click="-", keyboard_press="--", keyboard_hold="..")
        for mode, _data in response_data.groupby("mode"):
            response_colours = _data["task"].apply(
                lambda x: get_task_properties(x)["color"]
            )
            ax.vlines(
                _data["timestamp"],
                0.0,
                1.0,
                linewidth=1.0,
                linestyle=response_style[mode],
                colors=response_colours,
            )

    ax.set_xlim(start_time, finish_time)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def plot_failures(data, ax=None, alpha=1, ymin=0.05, ymax=0.95):
    if ax is None:
        ax = plt.gca()
    duration = data["finish_time"] - data["start_time"]
    ax.set_xlim(
        data["start_time"] - 0.1 * duration, data["finish_time"] + 0.1 * duration
    )
    ax.set_ylim(0, 1)
    fi = get_all_task_failure_intervals(data)
    offset = 0.0
    for task, group in fi.groupby("task"):
        color = get_task_properties(task)["color"]
        offset += 0.2
        intervals = group[["t1", "t2"]].to_numpy()
        intervals = merge_intervals(intervals)
        for interval in intervals:
            plot_interval_hatched(
                *interval,
                angle=45,
                density=int(duration * 4),
                ax=ax,
                alpha=1,
                color=color,
                offset=offset,
            )


def line_intersection(line1, line2):
    """
    Find the intersection of two lines.
    Each line is defined by a pair of points (x1, y1) and (x2, y2).
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Lines are parallel

    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator

    return px, py


def clip_line_to_rect(x_start, y_start, x_end, y_end, xmin, xmax, ymin, ymax):
    """
    Clip a line to the boundaries of a rectangle.
    """
    rect_lines = [
        (xmin, ymin, xmin, ymax),  # Left edge
        (xmin, ymax, xmax, ymax),  # Top edge
        (xmax, ymax, xmax, ymin),  # Right edge
        (xmax, ymin, xmin, ymin),  # Bottom edge
    ]

    clipped_line = []
    for rect_line in rect_lines:
        intersection = line_intersection((x_start, y_start, x_end, y_end), rect_line)
        if (
            intersection
            and xmin <= intersection[0] <= xmax
            and ymin <= intersection[1] <= ymax
        ):
            clipped_line.append(intersection)

    if len(clipped_line) < 2:
        return None  # Line does not intersect the rectangle

    return clipped_line[0], clipped_line[1]


def plot_interval_hatched(
    xmin,
    xmax,
    ymin=0.0,
    ymax=1.0,
    ax=None,
    color="black",
    angle=45,
    density=2,
    alpha=1.0,
    linewidth=1.0,
    offset=0.0,
):
    """
    Add hatched rectangle to an axis with an offset.

    NOTE: axis bounds must be set before hand!
    NOTE: this function is a bit tempremental when used with other plotting functionality as it depends on the axis bounds. I couldn't think of another way to do this while preserving "global" hatching. Density is also related to the axis lims, if the hatch is not showing try a very high density (e.g. 1000).

    Parameters:
    ax (matplotlib.axes.Axes): The axis to add the hatched region.
    xmin, xmax (float): The x-axis limits of the hatched region.
    ymin, ymax (float): The y-axis limits of the hatched region. Default 0.
    color (str): Color of the hatch lines.
    angle (float): Angle of the hatch lines in degrees.
    density (int): Number of hatch lines.
    alpha (float): Alpha of the hatch lines.
    linewidth (float): Width of the hatch lines.
    offset (float): Horizontal offset for the hatch lines. Useful for plotting overlapping hatches.
    """
    if ax is None:
        ax = plt.gca()

    angle_rad = np.radians(angle)
    slope = np.tan(angle_rad)

    # Get the axis limits to compute global hatch lines
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()

    # Calculate the diagonal length of the entire plotting area to ensure full coverage
    diagonal = np.hypot(ax_xlim[1] - ax_xlim[0], ax_ylim[1] - ax_ylim[0])

    # Determine the global line spacing based on density
    y_spacing = diagonal / density

    # Starting y-position (extended beyond the plot limits)
    y_start_global = ax_ylim[0] - diagonal + offset

    for i in range(2 * density):
        # Calculate the y-coordinates of the start and end points
        ys = y_start_global + i * y_spacing
        ye = ys + diagonal * slope

        # xs = xmin
        # xe = xmax
        xs = ax_xlim[0] - diagonal
        xe = ax_xlim[1] + diagonal

        clipped = clip_line_to_rect(xs, ys, xe, ye, xmin, xmax, ymin, ymax)
        if clipped:
            (xs, ys), (xe, ye) = clipped
            line = mlines.Line2D(
                [xs, xe], [ys, ye], color=color, alpha=alpha, linewidth=linewidth
            )
            ax.add_line(line)
