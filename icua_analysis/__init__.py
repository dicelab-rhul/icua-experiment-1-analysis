from . import preprocess

from .preprocess import load_tabularised, get_has_eyetracking
from .constants import (
    BACKGROUND_IMAGE,
    WINDOW_SIZE,
    TASK_COLORS,
    TASKS,
    TASK_NONE,
    TASK_RECTS,
    get_task_properties,
)

from .plot import (
    plot_intervals,
    plot_looking,
    plot_failures,
)

from .interval import (
    compute_intervals,
    merge_intervals,
    contained_in_intervals,
    get_non_overlapping_interval_indices,
)

from .dataframe import (
    get_system_task_data,
    get_fuel_task_data,
    get_tracking_task_data,
    get_click_data,
    get_keyboard_data,
    get_arrow_data,
    get_eyetracking_data,
    get_highlight_data,
    in_box,
    in_task_box,
    in_window_box,
    estimate_speed,
    zero_start_time,
    compute_task_failure_intervals,
    get_all_task_failure_intervals,
    get_looking_intervals,
    get_fixation_intervals,
    get_all_task_data,
    get_response_data,
)

from .performance_measures import default_performance, Performance


def melt_demographics(df_demo):
    return preprocess.melt_demographics_data(df_demo)


def load_demographics():
    return preprocess.load_demographics_data()
