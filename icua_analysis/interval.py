import itertools
import numpy as np
import pandas as pd
from types import SimpleNamespace


def get_non_overlapping_interval_indices(intervals, timestamps):
    # m and intervals must be sorted and intervals must be non-overlapping
    indices = -1 * np.ones(
        len(timestamps), dtype=int
    )  # Initialize with -1 for elements outside any interval
    interval_idx = 0

    for m_idx, timestamp in enumerate(timestamps):
        while interval_idx < len(intervals) and timestamp > intervals[interval_idx][1]:
            interval_idx += 1
        if (
            interval_idx < len(intervals)
            and intervals[interval_idx][0] <= timestamp < intervals[interval_idx][1]
        ):
            indices[m_idx] = interval_idx
    return indices


def merge_intervals(intervals):
    """Merges overlapping `intervals`. Non-overlapping intervals will be unchanged.

    Args:
        intervals (numpy.ndarray): intervals to merge.

    Returns:
        numpy.ndarray: merged intervals
    """
    if isinstance(intervals, list):
        assert all([y.shape[-1] == 2 for y in intervals])
        intervals = np.concatenate(intervals, axis=0)
    assert len(intervals.shape) == 2
    ts = intervals
    ts = ts[np.argsort(ts[:, 0])]  # sort by start times
    s, f = ts[:, 0], np.maximum.accumulate(ts[:, 1])
    v = np.ones(ts.shape[0] + 1, dtype=bool)
    v[1:-1] = s[1:] >= f[:-1]
    s, f = s[v[:-1]], f[v[1:]]
    return np.vstack([s, f]).T


def contained_in_intervals(intervals, timestamps):
    """Compute a binary array that corresponds to whether each timestamp in `timestamps` is contained within an interval in `intervals`.

    Args:
        intervals (numpy.ndarray (M,2)): intervals to check
        timestamps (numpy.ndarray (N,)): timestamps to check

    Returns:
        numpy.ndarray (N,): binary array representing whether timestamps are contained within any of the intervals.
    """
    # Create an empty binary array with the same length as timestamps
    result = np.zeros(len(timestamps), dtype=bool)
    for start, end in intervals:
        result |= (timestamps >= start) & (timestamps <= end)
    return result


def compute_intervals(binary, timestamps, start_time, finish_time, pad="finish"):
    binary = np.pad(
        binary.astype(np.uint8), (1, 1)
    )  # pad with zeros either side (ensures even index cardinality)
    _timestamps = np.pad(timestamps, (1, 1))  # pad with start/end time
    _timestamps[0] = start_time
    _timestamps[-1] = (
        finish_time if pad == "finish" else _timestamps[-2]
    )  # otherwise pad with the current value...
    y = np.pad(np.logical_xor(binary[:-1], binary[1:]), (1, 0))
    yi = np.arange(y.shape[0])[y]
    ts = _timestamps[yi].reshape(-1, 2)
    indices = get_non_overlapping_interval_indices(ts, timestamps)
    return SimpleNamespace(
        intervals=ts,
        indices=indices,
    )
