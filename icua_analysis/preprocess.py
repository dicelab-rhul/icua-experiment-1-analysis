"""Functions for preprocessing the raw event logs"""

from dataclasses import dataclass, asdict
from typing import List
from tqdm.auto import tqdm
import itertools
import re
import pathlib
import ast
import json
import pandas as pd

# widget names
SCALE_NAMES = ["Scale:0", "Scale:1", "Scale:2", "Scale:3"]
WARNINGLIGHT_NAMES = ["WarningLight:0", "WarningLight:1"]
FUEL_TANK_NAMES = ["FuelTank:A", "FuelTank:B"]
EYETRACKER_NAME = "EyeTracker:0"
EYETRACKERSTUB_NAME = "EyeTrackerStub"

SKIP_PARTICIPANTS = ["P00", "P03"]

# paths
RAW_DATA_PATH = "../data/raw_data/"
JSON_DATA_PATH = "../data/json_data/"


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


@dataclass
class LineData:
    indx: int
    timestamp: float
    event_src: str
    event_dst: str
    variables: dict

    @classmethod
    def from_line(cls, line):
        pattern = r"^(.*?):(\d+\.\d+) - \((.*?)\): ({.*?})$"
        match = re.match(pattern, line)
        if match is None:
            raise ValueError(f"Failed to match line: {line}")
        index = int(match.group(1))
        timestamp = float(match.group(2))
        event_src, event_dst = match.group(3).split("->")
        # 'cause' causes some issues because its not surrounded by quotes -_-
        variables = re.sub(r"'cause': (.*?{[^}]*})", r'"cause": "\1"', match.group(4))
        variables = ast.literal_eval(variables)
        cause = variables.get("cause", None)
        if cause:
            # keep only the index of the event that led to this event.
            variables["cause"] = match = re.match(pattern, variables["cause"]).group(1)
        elif "cause" in variables:
            del variables["cause"]
        return LineData(index, timestamp, event_src, event_dst, variables)

    @classmethod
    def get_start_time(cls, data):
        return data[0].timestamp

    @classmethod
    def get_finish_time(cls, data):
        return data[-1].timestamp

    @classmethod
    def findall_from_src(cls, data, event_src):
        return list(filter(lambda x: x.event_src == event_src, data))

    @classmethod
    def findall_from_dst(cls, data, event_dst):
        return list(filter(lambda x: x.event_dst == event_dst, data))

    @classmethod
    def findall_from_key_value(cls, data, key, value):
        return list(filter(lambda x: x.variables[key] == value, data))

    @classmethod
    def groupby_src(cls, data):
        return {
            k: list(sorted(v, key=lambda x: x.timestamp))
            for k, v in itertools.groupby(
                sorted(data, key=lambda x: x.event_src), key=lambda x: x.event_src
            )
        }

    @classmethod
    def groupby_dst(cls, data):
        return {
            k: list(sorted(v, key=lambda x: x.timestamp))
            for k, v in itertools.groupby(
                sorted(data, key=lambda x: x.event_dst), key=lambda x: x.event_dst
            )
        }

    @classmethod
    def contains_src(cls, data, event_src):
        try:
            next(filter(lambda x: x.event_src == event_src, data))  # exception if empty
            return True
        except:
            return False

    @classmethod
    def pack_variables(cls, data, *keys, sort_by="timestamp"):
        result = []
        sort_by = keys.index(sort_by)
        for line in data:
            result.append(
                [line.variables.get(k, asdict(line).get(k, None)) for k in keys]
            )
        return list(sorted(result, key=lambda v: v[sort_by]))

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(data):
        return LineData(**data)
