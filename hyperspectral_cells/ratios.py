import numpy as np
import pandas as pd
from scipy import stats


RATIO_NAMES = (
    "1612/(1530+1550)",
    "1530/(1530+1550)",
    "1612/(1612+1656)",
)


def summarize_ratios(peaks):
    if peaks.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = peaks.pivot_table(
        index=["cell_uid", "group", "condition", "time"],
        columns="forced_center_cm-1",
        values="peak_height",
        aggfunc="first",
    ).reset_index()

    def peak(center):
        return wide[center] if center in wide.columns else np.nan

    wide[RATIO_NAMES[0]] = peak(1612.0) / (peak(1530.0) + peak(1550.0))
    wide[RATIO_NAMES[1]] = peak(1530.0) / (peak(1530.0) + peak(1550.0))
    wide[RATIO_NAMES[2]] = peak(1612.0) / (peak(1612.0) + peak(1656.0))
    rows = []
    for (group, condition, time), frame in wide.groupby(
        ["group", "condition", "time"], dropna=False
    ):
        for ratio in RATIO_NAMES:
            values = frame[ratio].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "time": time,
                    "ratio": ratio,
                    "n_cells": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                }
            )
    tests = []
    for time, frame in wide.groupby("time", dropna=True):
        carbon = frame.loc[frame["condition"] == "Carbon 13"]
        if carbon.empty:
            continue
        for condition, comparison in frame.groupby("condition"):
            if condition == "Carbon 13":
                continue
            for ratio in RATIO_NAMES:
                reference = carbon[ratio].replace([np.inf, -np.inf], np.nan).dropna()
                values = comparison[ratio].replace([np.inf, -np.inf], np.nan).dropna()
                if len(reference) < 2 or len(values) < 2:
                    p_value = np.nan
                else:
                    p_value = float(stats.ttest_ind(values, reference, equal_var=False).pvalue)
                tests.append(
                    {
                        "time": time,
                        "group": comparison["group"].iloc[0],
                        "reference": f"Carbon 13 / {time}",
                        "ratio": ratio,
                        "p_value": p_value,
                    }
                )
    return wide, pd.DataFrame(tests)
