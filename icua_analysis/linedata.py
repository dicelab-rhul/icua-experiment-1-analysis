from dataclasses import dataclass, asdict
import itertools
import re
import ast


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
