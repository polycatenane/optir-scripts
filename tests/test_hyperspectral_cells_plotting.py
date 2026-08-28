import numpy as np
import pandas as pd

from hyperspectral_cells import plotting


RATIO = "1612/(1530+1550)"


def test_ratio_over_time_orders_hours_reuses_baseline_and_reports_counts():
    ratios = pd.DataFrame(
        {
            "condition": [
                "0 hour", "0 hour", "Control", "Control", "Control", "Control", "Acesulfame", "Acesulfame", "Acesulfame", "Acesulfame",
            ],
            "time": [
                "0 hour", "0 hour", "2 hour", "2 hour", "1 hour", "1 hour", "1 hour", "1 hour", "3 hour", "3 hour",
            ],
            RATIO: [1.0, 3.0, 6.0, np.inf, 4.0, 8.0, 2.0, 4.0, 8.0, 12.0],
        }
    )

    figure = plotting.ratio_over_time_figure(ratios, RATIO)

    assert figure is not None
    axis = figure.axes[0]
    assert axis.get_xticks().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Control (n=2/2/1/0)",
        "Carbon 13 (n=2/0/0/0)",
        "Acesulfame (n=2/2/0/2)",
        "Aspartame (n=2/0/0/0)",
        "Saccharin (n=2/0/0/0)",
        "Sucralose (n=2/0/0/0)",
    ]
    control = axis.containers[0].lines[0]
    acesulfame = axis.containers[2].lines[0]
    assert control.get_xdata().tolist() == [0.0, 1.0, 2.0]
    assert control.get_ydata().tolist() == [2.0, 6.0, 6.0]
    assert acesulfame.get_xdata().tolist() == [0.0, 1.0, 3.0]
    assert acesulfame.get_ydata().tolist() == [2.0, 3.0, 10.0]


def test_ratio_over_time_handles_singletons_and_bounded_ratio_axis():
    ratios = pd.DataFrame(
        {
            "condition": ["0 hour", "Control"],
            "time": ["0 hour", "1 hour"],
            "1530/(1530+1550)": [0.2, 0.6],
        }
    )

    figure = plotting.ratio_over_time_figure(ratios, "1530/(1530+1550)")

    assert figure is not None
    axis = figure.axes[0]
    assert axis.get_ylim() == (0.0, 1.0)
    assert axis.containers[0].lines[0].get_ydata().tolist() == [0.2, 0.6]


def test_ratio_over_time_returns_none_without_finite_baseline_or_ratio():
    ratios = pd.DataFrame(
        {
            "condition": ["Control"],
            "time": ["1 hour"],
            RATIO: [1.0],
        }
    )

    assert plotting.ratio_over_time_figure(ratios, RATIO) is None
    assert plotting.ratio_over_time_figure(ratios, "missing") is None
    assert plotting.ratio_over_time_figure(pd.DataFrame(), RATIO) is None


def test_ordering_and_empty_plot_helpers():
    ratios = pd.DataFrame(
        {
            "group": ["Control / 2 hour", "0 hour", "Carbon 13 / 1 hour"],
            "condition": ["Control", "0 hour", "Carbon 13"],
            "time": ["2 hour", "0 hour", "1 hour"],
            RATIO: [0.4, 0.2, 0.3],
        }
    )
    groups = {
        row.group: {"metadata": pd.DataFrame({"condition": [row.condition], "time": [row.time]})}
        for row in ratios.itertuples()
    }

    assert plotting.ordered_group_names(ratios) == ["0 hour", "Carbon 13 / 1 hour", "Control / 2 hour"]
    assert plotting.ordered_analysis_hours(groups) == ["0 hour", "1 hour", "2 hour"]
    assert plotting.ordered_previews([{"fov": "Control/2 hour/FOV1"}, {"fov": "0 hour/FOV1"}])[0]["fov"] == "0 hour/FOV1"
    assert plotting.ratio_boxplot_figure(pd.DataFrame(), RATIO, "1 hour") is None
    assert plotting.ratio_over_time_figure(pd.DataFrame(), RATIO) is None
