import numpy as np
import pandas as pd

from hyperspectral_cells.ratios import RATIO_NAMES, summarize_ratios


def peak_rows(cell_uid, group, condition, time, values):
    return [
        {"cell_uid": cell_uid, "group": group, "condition": condition, "time": time, "forced_center_cm-1": center, "peak_height": height}
        for center, height in values.items()
    ]


def test_summarize_ratios_preserves_formulas_and_ignores_infinite_values():
    peaks = pd.DataFrame(
        peak_rows("control", "Control / 1 hour", "Control", "1 hour", {1530.0: 2, 1550.0: 2, 1612.0: 3, 1656.0: 1})
        + peak_rows("zero", "Control / 1 hour", "Control", "1 hour", {1530.0: 0, 1550.0: 0, 1612.0: 1, 1656.0: 0})
    )

    wide, _ = summarize_ratios(peaks)

    values = wide.set_index("cell_uid")
    assert values.loc["control", list(RATIO_NAMES)].tolist() == [0.75, 0.5, 0.75]
    assert np.isinf(values.loc["zero", RATIO_NAMES[0]])


def test_summarize_ratios_handles_empty_and_welch_comparisons():
    assert all(frame.empty for frame in summarize_ratios(pd.DataFrame()))
    rows = []
    for condition, group, values in (
        ("Carbon 13", "Carbon 13 / 1 hour", [{1530.0: 1, 1550.0: 2, 1612.0: 2, 1656.0: 1}, {1530.0: 2, 1550.0: 1, 1612.0: 3, 1656.0: 2}]),
        ("Control", "Control / 1 hour", [{1530.0: 2, 1550.0: 1, 1612.0: 5, 1656.0: 1}, {1530.0: 1, 1550.0: 2, 1612.0: 2, 1656.0: 2}]),
    ):
        for index, peak_values in enumerate(values):
            rows.extend(peak_rows(f"{condition}-{index}", group, condition, "1 hour", peak_values))
    _, comparisons = summarize_ratios(pd.DataFrame(rows))

    assert len(comparisons) == len(RATIO_NAMES)
    assert comparisons["p_value"].notna().all()
