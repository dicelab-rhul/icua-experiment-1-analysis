import numpy as np
import matplotlib.image as mpimg

BACKGROUND_IMAGE = mpimg.imread("../data/background.png")  # background image


# widget names
PUMP = "Pump"
WARNINGLIGHT = "WarningLight"
SCALE = "Scale"

SCALE_NAMES = ["Scale:0", "Scale:1", "Scale:2", "Scale:3"]
WARNINGLIGHT_NAMES = ["WarningLight:0", "WarningLight:1"]
FUELTANK_NAMES = ["FuelTank:A", "FuelTank:B"]
EYETRACKER_NAME = "EyeTracker:0"
EYETRACKERSTUB_NAME = "EyeTrackerStub"

TARGET_NAME = "Target:0"
ARROW_NAME = "arrow_rotator_TEST"  # this is an unfortunate name...

TASK_NAME_FUEL = "resource management"
TASK_NAME_SYSTEM = "system monitoring"
TASK_NAME_TRACKING = "tracking"

# acceptable state information (from experiment configuration)
TARGET_ACCEPTABLE_DISTANCE = 50
FUEL_CAPACITY = 2000
FUEL_ACCEPTABLE_LEVEL = FUEL_CAPACITY / 2  # 1000
FUEL_ACCEPTABLE_RANGE = (
    FUEL_CAPACITY / 2 - FUEL_CAPACITY * 0.3,
    FUEL_CAPACITY / 2 + FUEL_CAPACITY * 0.3,
)
WARNINGLIGHT_ACCEPTABLE_STATES = [1, 0]
SCALE_ACCEPTABLE_STATE = 5

TASK_FUEL = "F"
TASK_SYSTEM = "S"
TASK_TRACKING = "T"

TRACKING_COLOR = "#4363d8"
FUEL_COLOR = "#3cb44b"
SYSTEM_COLOR = "#e6194B"

# window properties
WINDOW_SIZE = (800, 800)

TRACKING_WINDOW_PROPERTIES = {
    "position": np.array((351.25, 37.85)),
    "size": np.array((341.25, 341.25)),
    "color": TRACKING_COLOR,
    "name": TASK_NAME_TRACKING,
    "highlight_name": "Highlight:TrackingMonitor",
    # "data_fn": lambda dataset: get_tracking_task_data(dataset),
}
FUEL_WINDOW_PROPERTIES = {
    "position": np.array((253.75, 455.71428571428567)),
    "size": np.array((536.25, 334.2857142857143)),
    "color": FUEL_COLOR,
    "name": TASK_NAME_FUEL,
    "highlight_name": "Highlight:FuelMonitor",
    # "data_fn": lambda dataset: get_fuel_task_data(dataset),
}
SYSTEM_WINDOW_PROPERTIES = {
    "position": np.array((10.0, 37.857142857142854)),
    "size": np.array((243.75, 390.0)),
    "color": SYSTEM_COLOR,
    "name": TASK_NAME_SYSTEM,
    "highlight_name": "Highlight:SystemMonitor",
    # "data_fn": lambda dataset: get_system_monitor_task_data(dataset),
}


def get_task_properties(task):
    task = task[0].capitalize()
    if task == TASK_NONE:
        return {"name": "None", "color": "black"}
    return ALL_WINDOW_PROPERTIES[task]


ALL_WINDOW_PROPERTIES = {
    TASK_SYSTEM: SYSTEM_WINDOW_PROPERTIES,
    TASK_FUEL: FUEL_WINDOW_PROPERTIES,
    TASK_TRACKING: TRACKING_WINDOW_PROPERTIES,
}

TASK_NONE = "N"
TASKS = list(sorted(ALL_WINDOW_PROPERTIES.keys()))
TASK_COLORS = [x["color"] for x in dict(sorted(ALL_WINDOW_PROPERTIES.items())).values()]
TASK_RECTS = [
    (x["position"], x["size"], x["color"])
    for x in dict(sorted(ALL_WINDOW_PROPERTIES.items())).values()
]
