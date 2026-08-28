import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = (
    "Control",
    "Carbon 13",
    "Acesulfame",
    "Aspartame",
    "Saccharin",
    "Sucralose",
)
DEFAULT_GROUPS = ("0 hour", *CONDITIONS)


def hour_value(label: object) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", str(label))
    return float(match.group(1)) if match else None


def ratio_over_time_figure(ratios: pd.DataFrame, ratio_name: str):
    required = {"condition", "time", ratio_name}
    if ratios.empty or not required.issubset(ratios.columns):
        return None
    values = ratios.loc[:, ["condition", "time", ratio_name]].copy()
    values["hour"] = values["time"].map(hour_value)
    values[ratio_name] = pd.to_numeric(values[ratio_name], errors="coerce")
    values = values.loc[values["hour"].notna() & np.isfinite(values[ratio_name])]
    baseline = values.loc[values["condition"] == "0 hour"]
    if values.empty or baseline.empty:
        return None
    hours = sorted(values["hour"].unique())
    if not hours:
        return None
    figure, axis = plt.subplots(figsize=(9, 5))
    plotted = False
    for condition in CONDITIONS:
        condition_values = values.loc[values["condition"] == condition]
        summaries = []
        counts = []
        for hour in hours:
            samples = baseline.loc[baseline["hour"] == hour, ratio_name]
            if hour != 0:
                samples = condition_values.loc[condition_values["hour"] == hour, ratio_name]
            count = len(samples)
            counts.append(count)
            if not count:
                continue
            sem = samples.sem() if count > 1 else np.nan
            summaries.append((hour, samples.mean(), sem))
        if not summaries:
            continue
        plotted = True
        x, mean, sem = (np.asarray(values) for values in zip(*summaries, strict=True))
        axis.errorbar(
            x,
            mean,
            yerr=sem,
            marker="o",
            capsize=3,
            label=f"{condition} (n={'/'.join(map(str, counts))})",
        )
    if not plotted:
        plt.close(figure)
        return None
    axis.set_title(f"{ratio_name} over time")
    axis.set_xlabel("Hours")
    axis.set_ylabel("Mean cell-level ratio")
    axis.set_xticks(hours)
    axis.grid(True, axis="y", alpha=0.25)
    if ratio_name != "1612/(1530+1550)":
        axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    return figure


def group_sort_key(condition, time, group):
    if condition == "0 hour":
        return (0, 0, 0, group)
    match = re.search(r"(\d+)", str(time))
    time_order = int(match.group(1)) if match else 999
    condition_order = 0 if condition == "Carbon 13" else DEFAULT_GROUPS.index(condition) + 1 if condition in DEFAULT_GROUPS else 999
    return (1, time_order, condition_order, group)


def ordered_group_names(ratios):
    if ratios.empty or not {"group", "condition", "time"}.issubset(ratios.columns):
        return []
    groups = ratios.loc[:, ["group", "condition", "time"]].drop_duplicates()
    rows = groups.to_dict("records")
    return [
        row["group"]
        for row in sorted(
            rows,
            key=lambda row: group_sort_key(row["condition"], row["time"], row["group"]),
        )
    ]


def ordered_analysis_groups(groups):
    def key(group_name):
        metadata = groups[group_name]["metadata"]
        if metadata.empty:
            return (999, 999, 999, group_name)
        row = metadata.iloc[0]
        return group_sort_key(row["condition"], row["time"], group_name)

    return sorted(groups, key=key)


def ordered_analysis_hours(groups):
    hours = {
        hour
        for group in groups.values()
        for hour in group["metadata"]["time"].dropna().unique()
    }

    def key(hour):
        if hour == "0 hour":
            return (0, 0.0, str(hour))
        match = re.search(r"(\d+(?:\.\d+)?)", str(hour))
        return (1, float(match.group(1)) if match else float("inf"), str(hour))

    return sorted(hours, key=key)


def ordered_previews(previews):
    def key(preview):
        parts = preview["fov"].split("/")
        condition = parts[0]
        time = parts[1] if len(parts) > 1 else None
        return group_sort_key(condition, time, preview["fov"])

    return sorted(previews, key=key)


def segmentation_figure(preview):
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(preview["image"], cmap="gray")
    overlay = np.ma.masked_where(preview["masks"] == 0, preview["masks"])
    axis.imshow(overlay, cmap="nipy_spectral", alpha=0.35)
    axis.set_title(preview["fov"])
    axis.axis("off")
    figure.tight_layout()
    return figure


def stacked_spectra_figure(groups, selected_hour):
    selected_group_names = [
        group_name
        for group_name in ordered_analysis_groups(groups)
        if selected_hour in groups[group_name]["metadata"]["time"].unique()
    ]
    if not selected_group_names:
        return None
    figure, axis = plt.subplots(figsize=(10, max(4.5, 1.6 * len(selected_group_names))))
    offsets = []
    labels = []
    for index, group_name in enumerate(selected_group_names):
        group = groups[group_name]
        offset = 1.1 * index
        condition = group["metadata"]["condition"].iloc[0]
        axis.plot(group["x"], np.nanmean(group["spectra"], axis=0) + offset, color="black", label="Mean normalized AC" if index == 0 else None)
        axis.plot(group["x"], np.nanmean(group["fit_matrix"], axis=0) + offset, color="crimson", label="Mean Gaussian fit" if index == 0 else None)
        offsets.append(offset + 0.5)
        labels.append(condition)
    axis.set_title(f"Mean normalized AC spectra ({selected_hour})")
    axis.set_xlabel("Wavenumber (cm-1)")
    axis.set_ylabel("Normalized intensity (offset by condition)")
    axis.set_yticks(offsets, labels)
    axis.legend()
    figure.tight_layout()
    return figure


def ratio_boxplot_figure(ratios, ratio_name, selected_hour):
    if ratios.empty or ratio_name not in ratios.columns:
        return None
    hour_ratios = ratios.loc[ratios["time"] == selected_hour]
    if selected_hour == "0 hour":
        hour_ratios = hour_ratios.loc[hour_ratios["condition"] == "0 hour"]
    group_names = ordered_group_names(hour_ratios)
    figure, axis = plt.subplots(figsize=(max(9, 1.65 * len(group_names)), 5))
    series = []
    labels = []
    for group_name in group_names:
        values = hour_ratios.loc[hour_ratios["group"] == group_name, ratio_name].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if values.size:
            series.append(values)
            labels.append(hour_ratios.loc[hour_ratios["group"] == group_name, "condition"].iloc[0])
    if not series:
        plt.close(figure)
        return None
    positions = np.arange(1, len(series) + 1)
    axis.boxplot(series, positions=positions, showfliers=False, patch_artist=True)
    for position, values in zip(positions, series, strict=True):
        jitter = np.array([float(position)]) if values.size == 1 else position + np.linspace(-0.18, 0.18, num=values.size)
        axis.scatter(jitter, values, s=12, alpha=0.55, linewidths=0, zorder=3)
    axis.set_title(f"{ratio_name} ({selected_hour})")
    axis.set_ylabel("Cell-level ratio")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.grid(True, axis="y", alpha=0.25)
    if ratio_name != "1612/(1530+1550)":
        axis.set_ylim(0, 1)
    figure.tight_layout()
    return figure
