""" Convert preprocessed data into dataframes for analysis"""

from types import SimpleNamespace
import pandas as pd
import numpy as np
import itertools

from .linedata import LineData
from .constants import *
from .interval import *


def get_looking_intervals(data):
    """Gets the intervals during which a participant is looking at a given task (or no task). This includes both fixations and saccades.

    Args:
        data (pd.DataFrame): data for a given trial.

    Returns:
        pandas.DataFrame: intervals for where the participant is looking with the following columns ['t1', 't2', 'task'].
    """
    eye_data = data["eyetracking_data"]
    start_time, finish_time = data["start_time"], data["finish_time"]

    # if the eyetracker fails during the trial, the finish time may not be accurate. it should be one "tick" after the last recorded eye event
    assert len(eye_data) > 2
    finish_time = eye_data["timestamp"].iloc[-1] + (
        eye_data["timestamp"].iloc[-1] - eye_data["timestamp"].iloc[-2]
    )

    def _gen():
        for task in eye_data["in_task"].unique():
            eye_intervals = compute_intervals(
                eye_data["in_task"] == task,
                eye_data["timestamp"],
                start_time,
                finish_time,
            ).intervals
            yield pd.DataFrame(
                dict(
                    t1=eye_intervals[:, 0],
                    t2=eye_intervals[:, 1],
                    task=np.full(eye_intervals.shape[0], task),
                )
            )

    return pd.concat(_gen()).sort_values("t1").reset_index(drop=True)


def get_fixation_intervals(data):
    eye_data = data["eyetracking_data"]
    start_time, finish_time = data["start_time"], data["finish_time"]

    # if the eyetracker fails, the finish time may not be accurate. it should be one "tick" after the last eye event
    assert len(eye_data) > 2
    finish_time = eye_data["timestamp"].iloc[-1] + (
        eye_data["timestamp"].iloc[-1] - eye_data["timestamp"].iloc[-2]
    )

    # compute fixation intervals
    iresult = compute_intervals(
        eye_data["gaze"],
        eye_data["timestamp"],
        start_time,
        finish_time,
    )

    # below we will compute the task that is being fixated on, and check that this doesn't overlap multiple tasks
    intervals = iresult.intervals

    # mapping from eye events to interval index, use this to group eye events by the interval they belong to
    grouping = iresult.indices

    tasks = []
    # group to check "task" that is being fixated on. These should all be the same?
    for key, group in itertools.groupby(
        zip(grouping[grouping >= 0], eye_data[grouping >= 0].iterrows()), lambda x: x[0]
    ):
        group = np.array([x["in_task"] for _, (_, x) in group])
        unique = np.unique(group)
        # check that the group contains only a single task (or a single task + 'N'),
        # sometimes the fixation can be on the edge of the task
        if len(unique) == 1:
            tasks.append(unique[0])
        elif len(unique) == 2 and "N" in unique:
            # dont take 'N', this is because they are probably looking at the task,
            # there is eyetracker calibration error which we can account for a little here.
            unique = list(unique)
            unique.remove("N")
            tasks.append(unique[0])
        else:
            # something went wrong?
            print("---------------------------------")
            print(f"WARNING: Fixation across multiple tasks: {group} at interval{key}")
            print(intervals[key], key, len(intervals))
            print(grouping[np.arange(eye_data.shape[0])[grouping == key]])
            print(eye_data.iloc[np.arange(eye_data.shape[0])[grouping == key]])
            print("---------------------------------")
            tasks.append(
                unique[0]
            )  # just take the first one to prevent bugs later, but really we should think about this if it occurs
    return pd.DataFrame(
        dict(t1=intervals[:, 0], t2=intervals[:, 1], task=np.array(tasks))
    )


def get_response_data(keyboard_data, click_data, hold_action=False):
    """Merge keyboard and click data into a single `pandas.DataFrame`.
    Only the 'press' (and optionally 'hold') action(s) in the `keyboard_data` will be kept as these constitute a _response_.
    Args:
        keyboard_data (pandas.DataFrame): keyboard data frame
        click_data (pandas.DataFrame): click data frame
    Returns:
        (pandas.DataFrame) the resulting response data frame with the following columns ['timestamp', 'task', 'mode']
    """
    # merge keyboard and click data to get input data
    keyboard_response = keyboard_data["action"] == "press"
    keyboard_press_data = keyboard_data[keyboard_response][["timestamp"]].copy()
    keyboard_press_data["mode"] = np.full(len(keyboard_press_data), "keyboard_press")

    if hold_action:
        keyboard_response = keyboard_data["action"] == "hold"
        keyboard_hold_data = keyboard_data[keyboard_response][["timestamp"]].copy()
        keyboard_hold_data["mode"] = np.full(len(keyboard_hold_data), "keyboard_hold")
        keyboard_data = pd.concat([keyboard_press_data, keyboard_hold_data])
    else:
        keyboard_data = keyboard_press_data

    keyboard_data["task"] = np.full(len(keyboard_data), "T")
    click_data["mode"] = np.full(len(click_data), "mouse_click")

    input_data = pd.concat(
        [
            keyboard_data[["timestamp", "task", "mode"]],
            click_data[["timestamp", "task", "mode"]],
        ]
    )
    return input_data.sort_values("timestamp").reset_index(drop=True)


def get_all_task_data(data):
    return dict(T=data["tracking_data"], S=data["system_data"], F=data["fuel_data"])


def get_all_task_failure_intervals(data):
    start_time, finish_time = data["start_time"], data["finish_time"]
    data_tasks = get_all_task_data(data)
    # compute failure_intervals for each task
    fi = {
        name: compute_task_failure_intervals(data_task, start_time, finish_time)
        for name, data_task in data_tasks.items()
    }
    return pd.concat([df.assign(task=k) for k, df in fi.items()])


def zero_start_time(task_data, start_time, finish_time):
    """Sets the start time of provided start time to 0 based on `start_time`. This amounts to simply subtracting the start time from all timestamps.

    Args:
        task_data (List[pandas.DataFrame]): list of task data to zero start time.
        start_time (float): start time to use
        finish_time (float): finish time to use

    Returns:
        Tuple[List[pandas.DataFrame], float, float]: task_data where task_data['timestamp'] -= start_time, start_time = 0., finish_time - start_time
    """
    result_data = []
    for data in task_data:
        x = data.copy()
        x["timestamp"] -= start_time
        result_data.append(x)
    return result_data, 0.0, finish_time - start_time


def compute_task_failure_intervals(
    task_data: pd.DataFrame, start_time: float, finish_time: float
):
    """Computes failure intervals for the given task data. See also `compute_intervals`.

    Args:
        task_data (pd.DataFrame): task data to compute intervals for. Must have the following columns: ['timestamp', 'failure', 'component']
        start_time (float): start time of the trial
        finish_time (float): finish time of the trial

    Returns:
        pandas.DataFrame: a dataframe containing the failure intervals for the given task data, has the following columns: ['t1', 't2', 'component'].
    """
    # check for required columns
    assert "timestamp" in task_data.columns
    assert "failure" in task_data.columns
    assert "component" in task_data.columns
    df = pd.DataFrame(columns=["t1", "t2", "component"])
    # split by component
    for comp, data in task_data.groupby("component"):
        data = data.sort_values("timestamp").reset_index(drop=True)  # just in case
        ints = compute_intervals(
            data["failure"], data["timestamp"], start_time, finish_time
        ).intervals
        idf = pd.DataFrame(
            dict(t1=ints[:, 0], t2=ints[:, 1], component=np.full(ints.shape[0], comp))
        )
        df = pd.concat([df, idf])
    return df.reset_index(drop=True)


def estimate_speed(x, y, dt=1.0):
    """Estimates the speed from 2D coordinates using finite difference method.

    Args:
        x (numpy.ndarray): A numpy array of x-coordinates.
        y (numpy.ndarray: A numpy array of y-coordinates.
        dt (float): A float representing the time step between each coordinate pair. Defaults to 1.0.

    Returns:
        (numpy.ndarray): A numpy array of scalar speeds, each corresponding to the magnitude of a velocity vector at a time step.
    """
    # Calculate the differences between consecutive coordinates
    dx = np.diff(x)
    dy = np.diff(y)
    # Calculate the velocity in each direction
    vx = dx / dt
    vy = dy / dt
    # Combine the velocities into a single array of vectors
    velocities = np.column_stack((vx, vy))
    # Calculate the speed as the magnitude of the velocity vector
    speeds = np.sqrt(vx**2 + vy**2)
    return speeds


def in_box(x, y, position, size):
    """Compute a bool array representing whether the coordinates (x,y) are within a specific box as specified by its `position` and `size`.

    if x or y are > 1 dimensional they will be treated as a 1-dimensional array. The return array will have the same dimension as x and y.

    Args:
        x (numpy.ndarray): x coordinates
        y (numpy.ndarray): y coordinates
        position (Tuple[2]): position of the box
        size (Tuple[2]): size of the box

    Returns:
        np.ndarray (bool): bool array B that indicates when a corresponding coordinate (x[i],y[i]) -> B[i] is within the box.
    """
    assert tuple(x.shape) == tuple(y.shape)
    _xshape = x.shape
    x, y = x.flatten(), y.flatten()
    interval_x = (position[0], position[0] + size[0])
    interval_y = (position[1], position[1] + size[1])
    xok = np.logical_and(x > interval_x[0], x < interval_x[1])
    yok = np.logical_and(y > interval_y[0], y < interval_y[1])
    return np.logical_and(xok, yok).reshape(_xshape)


def in_task_box(x, y, task):
    """Computes a bool array representing whether the coordinates (x,y) are within a specific `task` box. See `in_box` for details.

    Args:
        x (numpy.ndarray): x coordinates
        y (numpy.ndarray): y coordinates
        task (str): name of the task

    Returns:
        np.ndarray (bool): bool array B that indicates when a corresponding coordinate (x[i],y[i]) -> B[i] is within the `task` box.
    """
    pos, size = (
        get_task_properties(task)["position"],
        get_task_properties(task)["size"],
    )
    return in_box(x, y, pos, size)


def in_window_box(x, y):
    """Computes a bool array representing whether the coordinates (x,y) are within the ICU window. See `in_box` for details.

    Args:
        x (numpy.ndarray): x coordinates
        y (numpy.ndarray): y coordinates

    Returns:
        np.ndarray (bool): bool array B that indicates when a corresponding coordinate (x[i],y[i]) -> B[i] is within the window.
    """
    pos = (0, 0)
    size = WINDOW_SIZE
    return in_box(x, y, pos, size)


def get_system_task_data(line_data):
    def get_warning_light_df(name, acceptable_state):
        data = np.array(
            LineData.pack_variables(
                LineData.findall_from_src(line_data, name), "timestamp", "value"
            )
        )
        component = np.full(data.shape[0], name)
        deviation = np.abs(data[:, 1] - acceptable_state)
        failure = data[:, 1] != acceptable_state
        return pd.DataFrame(
            dict(
                timestamp=data[:, 0],
                state=data[:, 1].astype(int),
                failure=failure,
                deviation=deviation,
                component=component,
            )
        )

    warning_light_dfs = [
        get_warning_light_df(name, acceptable_state)
        for name, acceptable_state in zip(
            WARNINGLIGHT_NAMES, WARNINGLIGHT_ACCEPTABLE_STATES
        )
    ]
    warning_light_dfs = pd.concat(warning_light_dfs)

    def get_scale_df(name, acceptable_state):
        data = np.array(
            LineData.pack_variables(
                LineData.findall_from_src(line_data, name), "timestamp", "value"
            )
        )
        component = np.full(data.shape[0], name)
        failure = data[:, 1] != acceptable_state
        deviation = np.abs(data[:, 1] - acceptable_state)
        return pd.DataFrame(
            dict(
                timestamp=data[:, 0],
                state=data[:, 1].astype(int),
                failure=failure,
                deviation=deviation,
                component=component,
            )
        )

    scale_dfs = [get_scale_df(name, SCALE_ACCEPTABLE_STATE) for name in SCALE_NAMES]
    scale_dfs = pd.concat(scale_dfs)
    return (
        pd.concat([scale_dfs, warning_light_dfs])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def get_fuel_task_data(line_data):
    # TODO doc string
    def _get_data(src, line_data):
        start_time, finish_time = line_data[0].timestamp, line_data[-1].timestamp
        line_data = LineData.findall_from_src(line_data, src)
        failure_data = LineData.findall_from_key_value(line_data, "label", "fuel")
        failure_data = np.array(
            LineData.pack_variables(failure_data, "timestamp", "acceptable")
        )
        failure_data = pd.DataFrame(
            dict(
                timestamp=failure_data[:, 0],
                failure=1 - failure_data[:, 1].astype(bool),
            )
        )
        # compute failure intervals
        start_time, finish_time = line_data[0].timestamp, line_data[-1].timestamp
        iresult = compute_intervals(
            failure_data["failure"], failure_data["timestamp"], start_time, finish_time
        )
        # based on failure intervals, compute the failure value for each row of level_data
        # level_data contains data about the fuel level of the tank
        # obviously the failure values should match up with what was set to acceptable in the config! otherwise sanity lost.
        level_data = LineData.findall_from_key_value(line_data, "label", "change")
        level_data = np.array(LineData.pack_variables(level_data, "timestamp", "value"))
        failure = contained_in_intervals(iresult.intervals, level_data[:, 0])
        deviation = np.abs(
            level_data[:, 1] - FUEL_ACCEPTABLE_LEVEL
        )  # deviation from the center of acceptable state...
        return pd.DataFrame(
            dict(
                timestamp=level_data[:, 0],
                state=level_data[:, 1],
                failure=failure,
                deviation=deviation,
                component=np.full(level_data.shape[0], src),
            )
        )

    tank_data = pd.concat([_get_data(src, line_data) for src in FUELTANK_NAMES])
    tank_data = tank_data.sort_values("timestamp").reset_index(drop=True)
    return tank_data


def get_tracking_task_data(line_data):
    # TODO doc string
    target_data = LineData.findall_from_src(line_data, TARGET_NAME)
    target_data = LineData.pack_variables(target_data, "timestamp", "x", "y")
    target_data = np.array(target_data)
    deviation = np.sqrt(np.sum(target_data[:, 1:] ** 2, axis=1))
    failure = deviation > TARGET_ACCEPTABLE_DISTANCE
    component = np.full(deviation.shape[0], TARGET_NAME)
    # TODO sort by timestamp?
    return pd.DataFrame(
        dict(
            timestamp=target_data[:, 0],
            x=target_data[:, 1],
            y=target_data[:, 2],
            failure=failure,
            deviation=deviation,
            component=component,
        )
    )


def get_eyetracking_data(dataset):
    # TODO docstring
    eye_data = LineData.pack_variables(
        LineData.findall_from_src(dataset, EYETRACKER_NAME),
        "timestamp",
        "label",
        "x",
        "y",
    )
    eye_data = np.array(eye_data)
    if eye_data.shape[0] == 0:
        return None  # no eyetracking data from this dataset! it is missing for some participants :(
    gi = (eye_data[:, 1] == "gaze").astype(bool)  # gaze = 1, saccade = 0
    t, x, y = (
        eye_data[:, 0].astype(np.float64),
        eye_data[:, 2].astype(np.float32),
        eye_data[:, 3].astype(np.float32),
    )
    # in task
    it = np.full_like(t, TASK_NONE, dtype=str)
    for task in TASKS:
        it[in_task_box(x, y, task)] = task
    # in window bounds
    iw = in_window_box(x, y)
    return pd.DataFrame(
        data=dict(timestamp=t, x=x, y=y, gaze=gi, in_task=it, in_window=iw)
    )


def get_highlight_data(dataset):
    # TODO docstring
    finish_time = LineData.get_finish_time(dataset)

    def get_data_from_source(src):
        data = np.array(
            LineData.pack_variables(
                LineData.findall_from_src(dataset, src), "timestamp", "value"
            )
        )
        data = data.reshape(data.shape[0], 2)  # in case there are no events
        if data.shape[0] % 2 != 0:
            # the session ended with a warning... add another event to match it (turn off at the end of session)
            data = np.concatenate([data, np.zeros((1, 2))])
            data[-1, 0] = finish_time
        return pd.DataFrame(dict(timestamp=data[:, 0], value=data[:, 1].astype(bool)))

    dfs = {
        k[0].capitalize(): get_data_from_source(v["highlight_name"])
        for k, v in ALL_WINDOW_PROPERTIES.items()
    }
    dfs_to_concat = [df.assign(task=key_name) for key_name, df in dfs.items()]
    # Concatenate the list of DataFrames
    return pd.concat(dfs_to_concat, ignore_index=True)


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
