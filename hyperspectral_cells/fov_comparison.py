from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from hyperspectral_cells import input, plotting


def exact_plane_path(paths: list[Path], wavenumber: float) -> Path:
    matches = [path for path in paths if input.parse_wavenumber(path) == wavenumber]
    if not matches:
        raise ValueError(f"No AC plane at {wavenumber:g} cm-1")
    if len(matches) > 1:
        raise ValueError(f"Duplicate AC planes at {wavenumber:g} cm-1")
    return matches[0]


def paired_dc_path(ac_path: Path) -> Path:
    if not ac_path.name.endswith("_AC.csv"):
        raise ValueError(f"Cannot derive paired DC filename from {ac_path.name}")
    dc_path = ac_path.with_name(ac_path.name.removesuffix("_AC.csv") + "_DC.csv")
    if not dc_path.is_file():
        raise ValueError(f"No paired DC plane for {ac_path.name}")
    return dc_path


def normalize_ac_plane(
    plane: np.ndarray,
    wavenumber: float,
    date: str,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if date not in profiles:
        raise ValueError(f"No processed IR-power profile matches acquisition date {date}")
    reference_x, reference_power = profiles[date]
    if not reference_x.min() <= wavenumber <= reference_x.max():
        raise ValueError(f"Wavenumber {wavenumber:g} cm-1 is outside the IR-power profile for {date}")
    power = np.interp(wavenumber, reference_x, reference_power)
    if not np.isfinite(power) or power <= 0:
        raise ValueError(f"IR-power value is not positive at {wavenumber:g} cm-1 for {date}")
    return plane / power


def load_fov_comparisons(
    root: Path,
    groups: tuple[str, ...],
    wavenumber_1: float,
    wavenumber_2: float,
) -> list[dict]:
    try:
        profiles = input.read_ir_profiles(root)
    except (FileNotFoundError, ValueError):
        profiles = {}

    rows = []
    for record in input.scan_ac_fovs(root, groups):
        if record["status"] != "pending":
            rows.append({**record, "load_error": record["message"]})
            continue
        row = {key: record[key] for key in ("group", "condition", "time", "relative_fov")}
        try:
            ac_1_path = exact_plane_path(record["files"], wavenumber_1)
            ac_2_path = exact_plane_path(record["files"], wavenumber_2)
            dc_path = paired_dc_path(ac_1_path)
            ac_1, ac_2, dc = (
                np.loadtxt(path, delimiter=",", dtype=np.float32)
                for path in (ac_1_path, ac_2_path, dc_path)
            )
            if any(plane.ndim != 2 for plane in (ac_1, ac_2, dc)):
                raise ValueError(f"Image plane is not two-dimensional: {record['relative_fov']}")
            if ac_1.shape != dc.shape or ac_2.shape != dc.shape:
                raise ValueError(f"Inconsistent image shapes in {record['relative_fov']}")
            rows.append(
                {
                    **row,
                    "status": "ok",
                    "load_error": "",
                    "dc": dc,
                    "ac_1": normalize_ac_plane(ac_1, wavenumber_1, input.parse_date(ac_1_path), profiles),
                    "ac_2": normalize_ac_plane(ac_2, wavenumber_2, input.parse_date(ac_2_path), profiles),
                    "wavenumber_1": input.parse_wavenumber(ac_1_path),
                    "wavenumber_2": input.parse_wavenumber(ac_2_path),
                    "pixel_size_um": input.parse_pixel_size_um(ac_1_path),
                    "acquisition_date_1": input.parse_date(ac_1_path),
                    "acquisition_date_2": input.parse_date(ac_2_path),
                }
            )
        except (OSError, ValueError) as error:
            rows.append({**row, "status": "error", "load_error": str(error)})
    return ordered_rows(rows)


def ordered_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: plotting.group_sort_key(
            row.get("condition"), row.get("time"), row.get("group", "")
        )
        + (row.get("relative_fov", str(row.get("fov_dir", ""))),),
    )


def ordered_times(rows: list[dict]) -> list[str]:
    times = {row.get("time") for row in rows if row.get("time") is not None}

    def key(time: str) -> tuple[int, float, str]:
        if time == "0 hour":
            return (0, 0.0, time)
        value = plotting.hour_value(time)
        return (1, value if value is not None else float("inf"), time)

    return sorted(times, key=key)


def status_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "FOV": row.get("relative_fov", str(row.get("fov_dir", ""))),
                "Condition": row.get("condition", ""),
                "Time": row.get("time", ""),
                "Status": row["status"],
                "Message": row.get("load_error", ""),
            }
            for row in rows
        ]
    )


def finite_normalization(images: list[np.ndarray]) -> Normalize:
    values = [image[np.isfinite(image)] for image in images]
    finite = np.concatenate([value for value in values if value.size]) if any(value.size for value in values) else np.array([])
    if not finite.size:
        return Normalize(0, 1)
    lower, upper = float(finite.min()), float(finite.max())
    return Normalize(lower, upper if upper > lower else lower + 1)


def selected_time_rows(rows: list[dict], selected_time: str) -> list[dict]:
    return ordered_rows(
        [row for row in rows if row.get("time") == selected_time and row["status"] == "ok"]
    )


def figure_control_selection(
    rows: list[dict], selected_time: str | None, selected_rows_per_figure: int | None
) -> tuple[list[str], str | None, list[int], int | None]:
    if not rows:
        return [], None, [], None
    times = ordered_times(rows)
    current_time = selected_time if selected_time in times else times[0]
    row_count_options = list(range(1, len(selected_time_rows(rows, current_time)) + 1))
    current_rows_per_figure = (
        selected_rows_per_figure
        if selected_rows_per_figure in row_count_options
        else min(4, len(row_count_options))
    )
    return times, current_time, row_count_options, current_rows_per_figure


def scale_bar_length_um(image_width_pixels: int, pixel_size_um: float) -> float:
    target_um = image_width_pixels * pixel_size_um * 0.2
    exponent = int(np.floor(np.log10(target_um)))
    candidates = [factor * 10**power for power in range(exponent - 1, exponent + 2) for factor in (1, 2, 5)]
    return min(candidates, key=lambda length: abs(length - target_um))


def add_scale_bar(axis, image: np.ndarray, pixel_size_um: float) -> None:
    height, width = image.shape
    length_um = scale_bar_length_um(width, pixel_size_um)
    length_pixels = length_um / pixel_size_um
    right = width - 0.5 - width * 0.04
    y = height - 0.5 - height * 0.07
    effects = [patheffects.withStroke(linewidth=3, foreground="black")]
    axis.plot(
        [right - length_pixels, right],
        [y, y],
        color="white",
        linewidth=3,
        solid_capstyle="butt",
        path_effects=effects,
    )
    axis.text(
        right - length_pixels / 2,
        y - height * 0.05,
        f"{length_um:g} µm",
        color="white",
        ha="center",
        va="bottom",
        path_effects=effects,
    )


def comparison_figure(
    rows: list[dict],
    normalizations: dict[str, Normalize] | None = None,
):
    if not rows:
        return None
    columns = ("dc", "ac_1", "ac_2")
    normalizations = normalizations or {
        column: finite_normalization([row[column] for row in rows]) for column in columns
    }
    figure = plt.figure(figsize=(16, max(3.5, 3.2 * len(rows))))
    grid = figure.add_gridspec(
        len(rows),
        10,
        width_ratios=(0.8, 0.12, 1, 1, 0.16, 0.055, 0.22, 1, 0.16, 0.055),
        wspace=0,
        hspace=0.08,
    )
    grid_columns = (2, 3, 7)
    images = {}
    for row_index, row in enumerate(rows):
        label = f"{row['condition']}\n{row['relative_fov']}"
        label_axis = figure.add_subplot(grid[row_index, 0])
        label_axis.text(1, 0.5, label, ha="right", va="center", transform=label_axis.transAxes)
        label_axis.axis("off")
        for column_index, column in enumerate(columns):
            axis = figure.add_subplot(grid[row_index, grid_columns[column_index]])
            image = axis.imshow(
                row[column],
                cmap="gray" if column == "dc" else "hot",
                norm=normalizations[column],
            )
            images.setdefault(column, image)
            if column == "dc":
                add_scale_bar(axis, row[column], row["pixel_size_um"])
            axis.axis("off")
            if row_index == 0:
                title = "DC" if column == "dc" else f"Normalized AC ({row[f'wavenumber_{column_index}']:g} cm-1)"
                axis.set_title(title)
    for column, colorbar_column in (("ac_1", 5), ("ac_2", 9)):
        colorbar = figure.colorbar(images[column], cax=figure.add_subplot(grid[:, colorbar_column]))
        colorbar.set_label("Power-normalized AC intensity")
    return figure


def comparison_figures(
    rows: list[dict], selected_time: str, rows_per_figure: int
) -> list:
    visible = selected_time_rows(rows, selected_time)
    normalizations = {
        column: finite_normalization([row[column] for row in visible])
        for column in ("dc", "ac_1", "ac_2")
    }
    return [
        comparison_figure(chunk, normalizations)
        for chunk in (visible[index : index + rows_per_figure] for index in range(0, len(visible), rows_per_figure))
    ]
