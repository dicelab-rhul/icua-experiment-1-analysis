""" Convert preprocessed data into dataframes for analysis"""

import pandas as pd
import numpy as np
from .preprocess import LineData

PUMP = "Pump"
WARNINGLIGHT = "WarningLight"
SCALE = "Scale"

TARGET_NAME = "Target:0"
ARROW_NAME = "arrow_rotator_TEST"  # this is an unfortunate name...

TASK_NAME_FUEL = "resource management"
TASK_NAME_SYSTEM = "system monitoring"
TASK_NAME_TRACKING = "tracking"


def get_arrow_data(dataset):
    """Get arrow data as a `DataFrame`

    Args:
        List[LineData] : data to get from

    Returns:
        DataFrame: keyboard data, columns = ['timestamp', 'angle_delta', 'angle']
    """
    line_data_arrow = LineData.findall_from_src(dataset, ARROW_NAME)
    data_arrow = np.array(
        LineData.pack_variables(line_data_arrow, "timestamp", "angle")
    ).reshape(len(line_data_arrow), 2)
    return pd.DataFrame(
        dict(
            timestamp=data_arrow[:, 0],
            angle_delta=data_arrow[:, 1],
            angle=data_arrow[:, 1].cumsum(),
        )
    )


def get_keyboard_data(dataset):
    """Get keyboard data as a `DataFrame`

    Args:
        List[LineData] : data to get from

    Returns:
        DataFrame: keyboard data, columns = ['timestamp', 'key', 'action']
    """
    line_data_keyboard = LineData.findall_from_src(dataset, "KeyHandler")
    data_keyboard = np.array(
        LineData.pack_variables(line_data_keyboard, "timestamp", "key", "action")
    )
    return pd.DataFrame(
        dict(
            timestamp=data_keyboard[:, 0],
            key=data_keyboard[:, 1],
            action=data_keyboard[:, 2],
        )
    )


def get_click_data(dataset):
    """Get click data as a dataframe.

    Args:
        List[LineData] : data to get from

    Returns:
        DataFrame: click data, columns = ['timestamp', 'dst', 'task', 'x', 'y', ]
    """
    line_data_mouse = LineData.findall_from_key_value(dataset, "label", "click")
    data_mouse = np.array(
        LineData.pack_variables(line_data_mouse, "timestamp", "event_dst", "x", "y")
    )
    df = pd.DataFrame(
        dict(timestamp=data_mouse[:, 0].astype(np.float64), dst=data_mouse[:, 1])
    )

    def get_task_from_component(x):
        if PUMP in x:
            return "F"
        elif WARNINGLIGHT in x or SCALE in x:
            return "S"
        else:
            raise ValueError(f"Unexpected component {x}")

    df["task"] = df["dst"].apply(get_task_from_component)
    df["x"] = data_mouse[:, 2].astype(int)
    df["y"] = data_mouse[:, 3].astype(int)
    return df
