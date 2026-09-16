from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pytest

from hyperspectral_cells import fov_comparison


def write_plane(path, value):
    np.savetxt(path, np.full((2, 2), value), delimiter=",")


def write_fov(root, group, fov, date, values, step_size_nm=150):
    directory = root / group / fov if group == "0 hour" else root / group / "1 hour" / fov
    directory.mkdir(parents=True)
    for wavenumber, (ac, dc) in values.items():
        step = (
            f"_step{step_size_nm}nmDwell20us400by400pts"
            if step_size_nm is not None
            else ""
        )
        stem = f"{date}{step}_{wavenumber}cm-1_scan"
        write_plane(directory / f"{stem}_AC.csv", ac)
        write_plane(directory / f"{stem}_DC.csv", dc)


def write_profile(root, date, powers):
    directory = root / "Power data Processed"
    directory.mkdir(exist_ok=True)
    pd.DataFrame({"Wavenumber_cm-1": list(powers), "Averaged_Power_mV": list(powers.values())}).to_csv(directory / f"power_{date}_processed.csv", index=False)


def comparison_row(index, **overrides):
    return {
        "status": "ok",
        "time": "1 hour",
        "condition": "Control",
        "relative_fov": f"Control/1 hour/FOV{index}",
        "dc": np.array([[1]]),
        "ac_1": np.array([[2]]),
        "ac_2": np.array([[3]]),
        "wavenumber_1": 1500,
        "wavenumber_2": 1600,
        "pixel_size_um": 0.15,
        **overrides,
    }


def test_loads_exact_planes_and_normalizes_by_date_matched_power(tmp_path):
    write_profile(tmp_path, "20260825", {1500: 2, 1600: 4})
    write_fov(tmp_path, "Control", "FOV1", "20260825", {1500: (8, 3), 1550: (15, 7), 1700: (50, 9)})
    write_fov(tmp_path, "Control", "FOV2", "20260825", {1500: (10, 4), 1550: (20, 8)}, 200)

    rows = fov_comparison.load_fov_comparisons(tmp_path, ("Control",), 1500, 1550)

    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert [row["wavenumber_1"] for row in rows] == [1500, 1500]
    assert [row["wavenumber_2"] for row in rows] == [1550, 1550]
    assert [row["pixel_size_um"] for row in rows] == [0.15, 0.2]
    np.testing.assert_allclose(rows[0]["ac_1"], 4)
    np.testing.assert_allclose(rows[0]["ac_2"], 5)
    np.testing.assert_allclose(rows[0]["dc"], 3)


def test_missing_exact_plane_and_invalid_power_are_per_fov_errors(tmp_path):
    write_profile(tmp_path, "20260825", {1500: 2, 1600: 0})
    write_fov(tmp_path, "Control", "FOV1", "20260825", {1500: (8, 3), 1600: (20, 7)})
    write_fov(tmp_path, "Control", "FOV2", "20260825", {1500: (8, 3)})

    rows = fov_comparison.load_fov_comparisons(tmp_path, ("Control",), 1500, 1600)

    assert [row["status"] for row in rows] == ["error", "error"]
    assert "not positive" in rows[0]["load_error"]
    assert "No AC plane" in rows[1]["load_error"]


def test_missing_pixel_size_is_a_per_fov_error(tmp_path):
    write_profile(tmp_path, "20260825", {1500: 2, 1600: 4})
    write_fov(
        tmp_path,
        "Control",
        "FOV1",
        "20260825",
        {1500: (8, 3), 1600: (20, 7)},
        step_size_nm=None,
    )

    rows = fov_comparison.load_fov_comparisons(tmp_path, ("Control",), 1500, 1600)

    assert rows[0]["status"] == "error"
    assert "Cannot read pixel size" in rows[0]["load_error"]


def test_preserves_multiple_fovs_and_orders_times(tmp_path):
    write_profile(tmp_path, "20260825", {1500: 2, 1600: 4})
    write_fov(tmp_path, "Control", "FOV2", "20260825", {1500: (8, 3), 1600: (20, 7)})
    write_fov(tmp_path, "Control", "FOV1", "20260825", {1500: (8, 3), 1600: (20, 7)})
    write_fov(tmp_path, "0 hour", "FOV1", "20260825", {1500: (8, 3), 1600: (20, 7)})

    rows = fov_comparison.load_fov_comparisons(tmp_path, ("Control", "0 hour"), 1500, 1600)

    assert [row["relative_fov"] for row in rows] == ["0 hour/FOV1", "Control/1 hour/FOV1", "Control/1 hour/FOV2"]
    assert fov_comparison.ordered_times(rows) == ["0 hour", "1 hour"]


def test_comparison_figures_share_norms_and_label_every_row_in_chunks():
    rows = [
        comparison_row(1, dc=np.array([[1, 2]]), ac_1=np.array([[3, 4]]), ac_2=np.array([[5, 6]])),
        comparison_row(2, dc=np.array([[0, 8]]), ac_1=np.array([[2, 10]]), ac_2=np.array([[4, 12]])),
        comparison_row(3),
    ]

    figures = fov_comparison.comparison_figures(rows, "1 hour", 2)
    figure = figures[0]

    assert len(figures) == 2
    figure.canvas.draw()
    image_axes = [axis for axis in figure.axes if axis.images]
    label_axes = [axis for axis in figure.axes if axis.texts and not axis.images]
    colorbar_axes = [axis for axis in figure.axes if not axis.images and not axis.texts]
    assert all(isinstance(axis.images[0].norm, Normalize) for axis in image_axes)
    second_image_axes = [axis for axis in figures[1].axes if axis.images]
    assert image_axes[0].images[0].norm is second_image_axes[0].images[0].norm
    assert image_axes[1].images[0].norm is second_image_axes[1].images[0].norm
    assert image_axes[2].images[0].norm is second_image_axes[2].images[0].norm
    assert [axis.images[0].get_cmap().name for axis in image_axes[:3]] == ["gray", "hot", "hot"]
    assert len(image_axes) == 6
    assert len(label_axes) == 2
    assert len(colorbar_axes) == 2
    assert all(label.get_position().x1 < image_axes[0].get_position().x0 for label in label_axes)
    assert [axis.texts[0].get_text() for axis in label_axes] == [
        f"{row['condition']}\n{row['relative_fov']}" for row in rows[:2]
    ]
    for item in figures:
        plt.close(item)


def test_comparison_figure_adds_scale_bars_to_dc_panels_only():
    rows = [
        comparison_row(
            index,
            dc=np.ones((40, 100)),
            ac_1=np.ones((40, 100)),
            ac_2=np.ones((40, 100)),
        )
        for index in range(1, 3)
    ]

    figure = fov_comparison.comparison_figure(rows)
    image_axes = [axis for axis in figure.axes if axis.images]
    dc_axes = [axis for axis in image_axes if axis.images[0].get_cmap().name == "gray"]
    ac_axes = [axis for axis in image_axes if axis.images[0].get_cmap().name == "hot"]

    assert all(len(axis.lines) == 1 for axis in dc_axes)
    assert all([text.get_text() for text in axis.texts] == ["2 µm"] for axis in dc_axes)
    assert all(
        np.isclose(np.diff(axis.lines[0].get_xdata())[0], 2 / 0.15)
        for axis in dc_axes
    )
    assert all(not axis.lines and not axis.texts for axis in ac_axes)
    plt.close(figure)


@pytest.mark.parametrize(("pixel_size_um", "expected"), [(0.05, 1), (0.1, 2), (0.15, 2), (0.25, 5)])
def test_scale_bar_length_uses_conventional_rounding(pixel_size_um, expected):
    assert (
        fov_comparison.scale_bar_length_um(100, pixel_size_um)
        == expected
    )


@pytest.mark.parametrize(
    ("selected_time", "selected_rows_per_figure", "expected"),
    [
        (None, None, (["0 hour", "1 hour"], "0 hour", [1], 1)),
        ("1 hour", 9, (["0 hour", "1 hour"], "1 hour", [1, 2, 3, 4, 5], 4)),
        ("missing", 2, (["0 hour", "1 hour"], "0 hour", [1], 1)),
    ],
)
def test_figure_control_selection_repairs_invalid_state(
    selected_time, selected_rows_per_figure, expected
):
    rows = [comparison_row(1, time="0 hour")] + [comparison_row(index) for index in range(2, 7)]

    assert fov_comparison.figure_control_selection(
        rows, selected_time, selected_rows_per_figure
    ) == expected


def test_figure_control_selection_keeps_empty_state_unset():
    assert fov_comparison.figure_control_selection([], "1 hour", 2) == ([], None, [], None)
