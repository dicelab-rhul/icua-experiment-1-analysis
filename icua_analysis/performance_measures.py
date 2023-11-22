import icua_analysis as ia
import pandas as pd
import numpy as np


def task_failure_length_mean(group, data):
    failure_intervals = ia.merge_intervals(data[["t1", "t2"]].to_numpy())
    return (failure_intervals[:, 1] - failure_intervals[:, 0]).mean()


def failure_length_mean(group, data):
    return (data["t2"] - data["t1"]).mean()


def failure_length_sum(group, data):
    failure_intervals = data[["t1", "t2"]].to_numpy()
    return (failure_intervals[:, 1] - failure_intervals[:, 0]).sum()


def task_failure_length_sum(group, data):
    failure_intervals = ia.merge_intervals(data[["t1", "t2"]].to_numpy())
    return (failure_intervals[:, 1] - failure_intervals[:, 0]).sum()


def task_failure_count(group, data):
    failure_intervals = ia.merge_intervals(data[["t1", "t2"]].to_numpy())
    return failure_intervals.shape[0]


def failure_count(group, data):
    return len(data)


# failure proportion requires additional information and so must be instantiated
# note that this is effectively the same as "failure_length_sum" as long as the trial duration is the same across trials/participants
class FailureProportion:
    def __init__(self, groupby):
        assert "participant" in groupby  # otherwise what is trial duration?
        assert "trial" in groupby  # otherwise what is trial duration?
        self.tds = Performance.get_trail_durations()
        self.pindex = groupby.index("participant")
        self.tindex = groupby.index("trial")
        self.__name__ = "failure_proportion"

    def __call__(self, group, data):  # metric
        # get trial duration based on trial/participant (they are all 180s but this will future proof things)
        trial_duration = self.tds[
            (self.tds["participant"] == group[self.pindex])
            & (self.tds["trial"] == group[self.tindex])
        ]["duration"].iloc[0]
        # merge intervals and compute failure proportion
        failure_intervals = ia.merge_intervals(data[["t1", "t2"]].to_numpy())
        failure_total = (failure_intervals[:, 1] - failure_intervals[:, 0]).sum()
        return failure_total / trial_duration


class Performance:
    @classmethod
    def get_all_failure_intervals(cls):
        def _dataframe_gen():
            tabuluar_dataset = ia.load_tabularised().has_eyetracking()
            for (participant, trial), data in tabuluar_dataset:
                fi = ia.get_all_task_failure_intervals(data)
                fi = fi.assign(participant=participant).assign(trial=trial)
                yield fi

        # these intervals are over all failures for all components in all tasks, all participants and all trials
        all_failure_intervals = pd.concat(_dataframe_gen())
        # add "difficulty" and "guidance" columns, then we can groupby them later :)
        all_failure_intervals["difficulty"] = np.array(["hard", "easy"])[
            all_failure_intervals["trial"].str.contains("A").to_numpy().astype(int)
        ]
        all_failure_intervals["guidance"] = all_failure_intervals["trial"].str.contains(
            "a"
        )
        return all_failure_intervals

    @classmethod
    def compute_performance(cls, metrics, groupby=["participant", "trial"]):
        afi = Performance.get_all_failure_intervals()
        metric_names = [m.__name__ for m in metrics]

        def _gen():
            for group, data in afi.groupby(groupby):
                yield (*group, *[metric(group, data) for metric in metrics])

        return pd.DataFrame(_gen(), columns=[*groupby, *metric_names])

    @classmethod
    def get_trail_durations(cls):
        tabuluar_dataset = ia.load_tabularised().has_eyetracking()

        def _gen():
            for (participant, trial), data in tabuluar_dataset:
                yield participant, trial, data["start_time"], data["finish_time"], data[
                    "finish_time"
                ] - data["start_time"]

        return pd.DataFrame(
            _gen(),
            columns=["participant", "trial", "start_time", "finish_time", "duration"],
        )


def default_performance():
    groupby = ["participant", "trial"]
    metrics = [
        FailureProportion(groupby),
        task_failure_length_mean,
        failure_length_mean,
        task_failure_length_sum,
        failure_length_sum,
        task_failure_count,
        failure_count,
    ]
    return Performance.compute_performance(metrics)


if __name__ == "__main__":
    # example of computing performance metrics. group by can also include "difficulty", "task", "guidance"
    groupby = ["participant", "trial"]
    metrics = [
        FailureProportion(groupby),
        task_failure_length_mean,
        failure_length_mean,
        task_failure_length_sum,
        failure_length_sum,
        task_failure_count,
        failure_count,
    ]

    performance = Performance.compute_performance(metrics)
    print(performance)
