from .constants import *
from .interval import *
from .preprocess import load_tabularised


def get_guidance_stats():
    """Compute guidance statistics for each participant and collect into a dataframe.


    Returns:
        (pandas.DataFrame): a data frame with the following columns [
            "participant",
            "trial",
            "task",
            "guidance_cumulative",
            "guidance_count",
        ]
    """
    result = pd.DataFrame(
        columns=[
            "participant",
            "trial",
            "task",
            "guidance_cumulative",
            "guidance_count",
        ]
    )
    for trial, par_data in (
        load_tabularised().has_eyetracking().has_guidance().groupby_trial()
    ):
        for par, data in par_data.items():
            guidance_data = data.get(
                "highlight_data", pd.DataFrame(columns=["timestamp", "value", "task"])
            )  # get guidance data (it is not always present, this means no guidance was given)
            guidance_data["timestamp"] -= data["start_time"]
            for task in TASKS:
                tdf = guidance_data[guidance_data["task"] == task]
                tintervals = compute_intervals(
                    tdf["value"].to_numpy(),
                    tdf["timestamp"].to_numpy(),
                    0,
                    data["finish_time"] - data["start_time"],
                ).intervals
                result.loc[len(result)] = [
                    par,
                    trial,
                    task,
                    (tintervals[:, 1] - tintervals[:, 0]).sum(),
                    len(tintervals),
                ]
    return result
