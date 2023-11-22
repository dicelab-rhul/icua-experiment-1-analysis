"""Functions for preprocessing the raw event logs"""

from dataclasses import dataclass, asdict
from typing import List
from tqdm.auto import tqdm
import pathlib
import json
import pandas as pd
import itertools

from .constants import *

from .dataframe import (
    get_click_data,
    get_keyboard_data,
    get_arrow_data,
    get_eyetracking_data,
    get_highlight_data,
    get_tracking_task_data,
    get_system_task_data,
    get_fuel_task_data,
)

from .linedata import LineData

SKIP_PARTICIPANTS = ["P00", "P03"]

# paths
RAW_DATA_PATH = "../data/raw_data/"
JSON_DATA_PATH = "../data/json_data/"
TAB_DATA_PATH = "../data/tabular_data/"


class TabularisedData:
    def __init__(self, data):
        super().__init__()
        self.data = data
        self._has_eyetracking = False
        self._only_easy = False
        self._only_hard = False
        self._only_guidance = False
        self._keys = None

    def groupby_participant(self):
        trial_order = ["icuA", "icuaA", "icuB", "icuaB"]
        trial_sorting_fun = lambda x: trial_order.index(x[0][1])
        # this will yield (par, group)
        for par, group in itertools.groupby(
            sorted(self, key=lambda x: x[0][0]), lambda x: x[0][0]
        ):
            yield par, {
                trial: data for (_, trial), data in sorted(group, key=trial_sorting_fun)
            }

    def groupby_trial(self):
        trial_order = ["icuA", "icuaA", "icuB", "icuaB"]
        trial_sorting_fun = lambda x: trial_order.index(x[0][1])
        trial_grouping_fun = lambda x: x[0][1]
        # this will yield (trial, group)
        for trial, group in itertools.groupby(
            sorted(self, key=trial_sorting_fun), trial_grouping_fun
        ):
            yield trial, {par: data for (par, _), data in group}

    def has_eyetracking(self) -> "TabularisedData":
        self._has_eyetracking = True
        return self

    def is_easy(self) -> "TabularisedData":
        self._only_easy = True
        return self

    def is_hard(self) -> "TabularisedData":
        self._only_hard = True
        return self

    def has_guidance(self) -> "TabularisedData":
        self._only_guidance = True
        return self

    def reset_filters(self) -> "TabularisedData":
        self._has_eyetracking = False
        self._only_easy = False
        self._only_hard = False
        return self

    def __iter__(self):
        for par, x in self.data.items():
            if self._has_eyetracking:
                has_eyetracking = all(("eyetracking_data" in y) for y in x.values())
                if not has_eyetracking:
                    continue
            for exp, y in x.items():
                if self._only_easy and not "A" in exp:
                    continue
                if self._only_hard and not "B" in exp:
                    continue
                if self._only_guidance and not "a" in exp:
                    continue
                yield (par, exp), y if self._keys is None else {
                    y[k] for k in self._keys
                }


def get_has_eyetracking():
    """Get all participants who have valid eyetracking data.

    Returns:
        List[str]: participants with valid eyetracking data.
    """
    return [
        par for par, _ in load_tabularised().has_eyetracking().groupby_participant()
    ]


def load_tabularised(path=TAB_DATA_PATH, skip=[]):
    def _load(base_path):
        base_path = pathlib.Path(base_path)
        data = {}
        meta_path = pathlib.Path(base_path, "meta.json")
        if meta_path.exists():
            with open(meta_path, "r") as file:
                meta_data = json.load(file)
                data.update(meta_data)
        for path in base_path.iterdir():
            if any([(path in ig) for ig in skip]):
                continue
            if path.is_dir():
                data[path.name] = _load(path)
            elif path.suffix == ".csv":  # Handle DataFrame files
                key = path.name.split(".")[0]  # Remove '.csv' extension
                data[key] = pd.read_csv(path)
        return dict(sorted(data.items()))

    return TabularisedData(_load(path))


def save_tabularised(path=TAB_DATA_PATH):
    base_path = pathlib.Path(path)
    if not base_path.exists():
        base_path.mkdir()

    for file, data in tqdm(load_from_json()):
        parexp = file.name.split(".")[0]
        par, exp = parexp[:3], parexp[3:]
        path = pathlib.Path(base_path, par, exp)
        start, finish = get_start_and_finish_time(data)
        data = {
            "start_time": start,
            "finish_time": finish,
            "click_data": get_click_data(data),
            "keyboard_data": get_keyboard_data(data),
            "arrow_data": get_arrow_data(data),
            "eyetracking_data": get_eyetracking_data(data),
            "highlight_data": get_highlight_data(data),
            "tracking_data": get_tracking_task_data(data),
            "fuel_data": get_fuel_task_data(data),
            "system_data": get_system_task_data(data),
        }
        _save_nested_dict(data, path)


# save tabularised data that is part of a dict (see `save_tabularised`)
def _save_nested_dict(data, path):
    if data is None:
        return
    path = pathlib.Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    meta_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            _save_nested_dict(value, pathlib.Path(path, path, str(key)))
        elif isinstance(value, pd.DataFrame):  # Handle DataFrame objects
            if len(value) > 0:
                value.to_csv(str(pathlib.Path(path, f"{key}.csv")), index=False)
        elif not value is None:
            meta_data[key] = value
    if meta_data:
        with open(str(pathlib.Path(path, "meta.json")), "w") as file:
            json.dump(meta_data, file)


def get_start_and_finish_time(data):
    return data[0].timestamp, data[-1].timestamp


def preprocess_and_save_as_json(
    path: str = RAW_DATA_PATH,
    json_path: str = JSON_DATA_PATH,
    skip: List[str] = SKIP_PARTICIPANTS,
):
    """preprocess raw event data and save it as json data.

    Args:
        path (str, optional): path to raw event data. Defaults to RAW_DATA_PATH.
        json_path (str, optional): path to save json data. Defaults to JSON_DATA_PATH.
        skip (List[str], optional): participants to skip. Defaults to SKIP_PARTICIPANTS.
    """

    def save_as_json(file: str, data: List[LineData]):
        path = pathlib.Path(json_path, file)
        if not path.parent.exists():
            path.parent.mkdir()
        with open(str(path), "w") as f:
            json.dump([line.as_dict() for line in data], f)

    for file, data in preprocess(path=path, skip=skip):
        save_as_json(file.name.replace(".txt", ".json"), data)


def load_from_json(path=JSON_DATA_PATH):
    """Loads data from preprocessed json files (see `preprocess_and_save_as_json`).

    Args:
        path (str, optional): path to json data. Defaults to JSON_DATA_PATH.

    Yields:
        Tuple[str, List[LineData]]): loaded data as collections of `LineData` (file, data).
    """
    path = pathlib.Path(path)
    # download(?) and generate a dataset from a given trial log file.
    files = list(
        sorted([f for f in path.iterdir() if f.suffix == ".json"], key=lambda f: f.name)
    )
    for file in files:
        with open(file, "r") as f:
            yield file, [LineData(**x) for x in json.load(f)]


def preprocess(path=RAW_DATA_PATH, skip=SKIP_PARTICIPANTS):
    """Preprocess the raw event data to produce collections of `LineData`.

        Note: P03 did not complete all of the trials (one was repeated, oops!)
        Note: P00 is a test run and not an actual participant.

    Args:
        path (str, optional): path to raw data. Defaults to RAW_DATA_PATH.
        skip (List[str], optional): participants to skip.

    Yields:
        Tuple[str, List[LineData]]: collection of line data for a given file as tuples: (file, data)
    """
    path = pathlib.Path(path)
    # download(?) and generate a dataset from a given trial log file.
    files = list(
        sorted([f for f in path.iterdir() if f.suffix == ".txt"], key=lambda f: f.name)
    )

    def data_generator(file):
        with open(file, "r") as f:
            for line in f:
                yield LineData.from_line(line)

    def correct_eyetracking_timestamps(data):
        def eye_tracking_transitions():
            for i in range(len(data)):
                if (
                    data[i].event_src == EYETRACKER_NAME
                    and data[i - 1].event_src != EYETRACKER_NAME
                ):
                    # provide the eyetracking event timestamp and the unix timestamp immediately prior to it
                    yield data[i].timestamp, data[i - 1].timestamp

        titer = eye_tracking_transitions()
        has_eyetracking_data = next(titer, None)
        if not has_eyetracking_data:
            # this means the stub eyetracker was used and the timestamps do not need correcting
            return data
        # use the second transition, this means the eyetracking timestamp will be off by at most the eyetrackers sampling rate
        t2, tu2 = next(titer)
        # correct the eyetracking timestamp
        for line in data:
            if line.event_src == EYETRACKER_NAME:
                line.timestamp = line.timestamp - t2 + tu2
        return data

    for file in tqdm(files, desc="preprocessing log files..."):
        if any([s in str(file) for s in skip]):
            print(f"skipping file {file}")
            continue
        data = [line for line in data_generator(file)]
        data = correct_eyetracking_timestamps(data)
        data = list(sorted(data, key=lambda x: x.timestamp))
        yield file, data


def melt_demographics_data(df_demo):
    # Melting for T columns
    melted_T = df_demo.melt(
        id_vars=["participant", "age", "gender"],
        value_vars=["T0", "T1", "T2", "T3"],
        var_name="_",
        value_name="trial",
    )

    # Melting for D columns
    melted_D = df_demo.melt(
        id_vars=["participant", "age", "gender"],
        value_vars=["D0", "D1", "D2", "D3"],
        var_name="_",
        value_name="estimated_difficulty",
    )

    # Adding a custom identifier to align both melts
    melted_T["time"] = melted_T["_"].str[1]
    melted_D["time"] = melted_D["_"].str[1]

    # Merging the melted DataFrames
    result = pd.merge(
        melted_T.drop(columns="_"),
        melted_D.drop(columns="_"),
        on=["participant", "age", "gender", "time"],
    )
    result = result[
        [
            "participant",
            "trial",
            "age",
            "gender",
            "time",
            "estimated_difficulty",
        ]
    ]
    return result


def load_demographics_data(skip=["P00", "P03"]):
    df = pd.read_excel("../data/raw_data/demographics.xlsx")
    df = df[
        [
            "Participant Number",
            "Age",
            "Gender",
            "Easy icu A",
            "Easy icua A",
            "Hard icu B",
            "Hard icua B",
            "Scores",
            "Scores.1",
            "Scores.2",
            "Scores.3",
        ]
    ]
    df = df.rename(
        columns={
            "Participant Number": "participant",
            "Age": "age",
            "Gender": "gender",
            "Easy icu A": "T0",
            "Easy icua A": "T1",
            "Hard icu B": "T2",
            "Hard icua B": "T3",
            "Scores": "D0",
            "Scores.1": "D1",
            "Scores.2": "D2",
            "Scores.3": "D3",
        }
    )
    df = df.applymap(
        lambda x: x.replace(" ", "").replace("'", "") if isinstance(x, str) else x
    )
    df = df[~df["participant"].isin(skip)]
    return df
