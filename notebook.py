# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "accelerate>=1.14.0",
#     "cellpose==3.1.1.3",
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.1",
#     "numpy>=2.5.2",
#     "pandas==2.3.3",
#     "scipy>=1.18.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # O-PTIR hyperspectral analysis

    ## Expected data layout

    Set `DATA_ROOT` to the directory containing the processed power data and one
    directory per experimental condition:

    ```text
    data-root/
    ├── Power data Processed/
    │   └── power_20260825_processed.csv
    ├── 0 hour/
    │   └── FOV1/
    │       └── 20260825_1612cm-1_scan_AC.csv
    └── Control/
        └── 1 hour/
            └── FOV1/
                └── 20260825_1612cm-1_scan_AC.csv
    ```

    The selected group names must match the directories directly below the data root.
    AC image files are discovered recursively from files ending in `_AC.csv`. Each filename
    must contain an eight-digit acquisition date and a wavenumber in the form
    `_1612cm-1_`. Images in the same FOV must be equally sized 2D CSV arrays with unique
    wavenumbers. Each file contains the image measured at one wavenumber.

    The notebook pairs AC data with power data by the date in each filename. For example,
    AC files beginning with `20260825` use the power data CSV containing `20260825` in its
    name. A power data file is ignored when there is no AC data from the same date. If AC
    data has no matching power data file, that FOV is reported as an error.

    Power data CSVs must contain `Wavenumber_cm-1` and `Averaged_Power_mV` columns.

    ## Processing

    For each FOV, the notebook:

    1. Sorts the AC images by wavenumber and combines them into a spectral image stack.
    2. Averages the selected spectral window and segments that image with Cellpose.
    3. Averages the spectrum inside each segmentation mask.
    4. Corrects each spectrum with the matching power data.
    5. Filters and normalizes the spectra, fits constrained Gaussian peaks, and
       calculates the reported peak ratios.
    """)
    return


@app.cell
def _():
    from collections import defaultdict
    from functools import reduce
    from pathlib import Path

    from accelerate import Accelerator
    from cellpose import models
    from hyperspectral_cells import fitting, input, plotting, ratios
    import marimo as mo
    import numpy as np
    import pandas as pd

    DATA_ROOT = Path(
        "/var/home/praxis/Documents/Bai/Data/20260828 Artificial Sweetener"
    ) / "2026-08-25 and 2026-08-26"
    DEFAULT_GROUPS = (
        "0 hour",
        "Control",
        "Acesulfame",
        "Aspartame",
        "Carbon 13",
        "Saccharin",
        "Sucralose",
    )
    return (
        Accelerator,
        DATA_ROOT,
        DEFAULT_GROUPS,
        Path,
        defaultdict,
        fitting,
        input,
        mo,
        models,
        np,
        pd,
        plotting,
        ratios,
        reduce,
    )


@app.cell
def _(
    Accelerator,
    defaultdict,
    fitting,
    input,
    models,
    np,
    pd,
    ratios,
    reduce,
):
    def run_ac_analysis(settings):
        root = settings["data_root"]
        if not root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        profiles = input.read_ir_profiles(root)
        accelerator = Accelerator()
        device = accelerator.device
        model = models.CellposeModel(
            gpu=device.type != "cpu",
            device=device,
            model_type=settings["model_type"],
        )
        records = input.scan_ac_fovs(root, settings["groups"])
        fov_summary = []
        group_entries = defaultdict(list)
        previews = []

        for record in records:
            if record["status"] != "pending":
                fov_summary.append(
                    {"fov": str(record["fov_dir"]), "status": record["status"], "message": record["message"]}
                )
                continue
            try:
                ac_cube, x = input.read_ac_cube(record["files"])
                window = (x >= settings["segment_low"]) & (x <= settings["segment_high"])
                if not np.any(window):
                    raise ValueError(
                        f"No AC planes between {settings['segment_low']:g} and {settings['segment_high']:g} cm-1"
                    )
                segmentation_image = np.nanmean(ac_cube[:, :, window], axis=-1)
                masks, _, _ = model.eval(
                    segmentation_image,
                    channels=[0, 0],
                    channel_axis=None,
                    diameter=settings["diameter"],
                    flow_threshold=settings["flow_threshold"],
                    cellprob_threshold=settings["cellprob_threshold"],
                )
                raw_spectra, labels, pixel_counts = input.extract_label_spectra(ac_cube, masks)
                previews.append(
                    {
                        "fov": record["relative_fov"],
                        "image": segmentation_image,
                        "masks": masks,
                    }
                )
                if raw_spectra.size == 0:
                    fov_summary.append(
                        {
                            "fov": record["relative_fov"],
                            "group": record["group"],
                            "status": "no_masks",
                            "n_masks": 0,
                            "message": "Cellpose returned no non-background labels",
                        }
                    )
                    continue
                date = record["files"] and record["files"][0] and record["files"][0].name[:8]
                x_supported, corrected = input.align_to_ir_power(x, raw_spectra, date, profiles)
                metadata = pd.DataFrame(
                    {
                        "cell_uid": [f"{record['relative_fov']}::label-{label}" for label in labels],
                        "mask_label": labels,
                        "pixel_count": pixel_counts,
                        "fov": record["relative_fov"],
                        "group": record["group"],
                        "condition": record["condition"],
                        "time": record["time"],
                        "acquisition_date": date,
                    }
                )
                group_entries[record["group"]].append(
                    {
                        "x": x_supported,
                        "corrected": corrected,
                        "raw": raw_spectra[:, np.isin(x, x_supported)],
                        "metadata": metadata,
                        "condition": record["condition"],
                        "time": record["time"],
                    }
                )
                fov_summary.append(
                    {
                        "fov": record["relative_fov"],
                        "group": record["group"],
                        "status": "ok",
                        "n_masks": int(len(labels)),
                        "message": "",
                    }
                )
            except Exception as error:
                fov_summary.append(
                    {
                        "fov": record["relative_fov"],
                        "group": record["group"],
                        "status": "error",
                        "n_masks": 0,
                        "message": str(error),
                    }
                )

        groups = {}
        peak_tables = []
        fit_summaries = []
        for group, entries in group_entries.items():
            shared_x = reduce(np.intersect1d, [entry["x"] for entry in entries])
            if shared_x.size < 3:
                continue
            corrected = np.vstack(
                [entry["corrected"][:, np.isin(entry["x"], shared_x)] for entry in entries]
            )
            raw = np.vstack([entry["raw"][:, np.isin(entry["x"], shared_x)] for entry in entries])
            metadata = pd.concat([entry["metadata"] for entry in entries], ignore_index=True)
            keep_count = max(1, int(np.ceil(corrected.shape[0] * 0.90)))
            keep = np.argsort(np.nanmax(raw, axis=1))[::-1][:keep_count]
            corrected = corrected[keep]
            metadata = metadata.iloc[keep].reset_index(drop=True)
            row_max = np.nanmax(corrected, axis=1, keepdims=True)
            normalized = corrected / np.where(row_max == 0, np.nan, row_max)
            valid = np.all(np.isfinite(normalized), axis=1)
            normalized = normalized[valid]
            metadata = metadata.iloc[np.flatnonzero(valid)].reset_index(drop=True)
            if normalized.size == 0:
                continue
            fitted = fitting.fit_group_spectra(shared_x, normalized, metadata)
            groups[group] = {
                "x": shared_x,
                "spectra": normalized,
                "metadata": metadata,
                **fitted,
            }
            if not fitted["peaks"].empty:
                peak_tables.append(fitted["peaks"])
            fit_summaries.append(fitted["fit_summary"])

        peaks = pd.concat(peak_tables, ignore_index=True) if peak_tables else pd.DataFrame()
        fit_summary = pd.concat(fit_summaries, ignore_index=True) if fit_summaries else pd.DataFrame()
        ratio_values, p_values = ratios.summarize_ratios(peaks)
        return {
            "device": str(device),
            "fov_summary": pd.DataFrame(fov_summary),
            "groups": groups,
            "peaks": peaks,
            "fit_summary": fit_summary,
            "ratios": ratio_values,
            "p_values": p_values,
            "previews": previews,
        }

    return (run_ac_analysis,)


@app.cell
def _(DATA_ROOT, DEFAULT_GROUPS, mo):
    data_root = mo.ui.text(value=str(DATA_ROOT), label="AC data root")
    selected_groups = mo.ui.multiselect(
        options=list(DEFAULT_GROUPS), value=list(DEFAULT_GROUPS), label="Validated groups"
    )
    model_type = mo.ui.dropdown(
        options=["bact_fluor_cp3", "bact_phase_cp3", "cyto2_cp3"],
        value="bact_fluor_cp3",
        label="Cellpose model",
    )
    segment_low = mo.ui.number(start=1000, stop=1800, step=5, value=1600, label="Segmentation low cm-1")
    segment_high = mo.ui.number(start=1000, stop=1800, step=5, value=1700, label="Segmentation high cm-1")
    diameter = mo.ui.number(start=0, stop=500, step=1, value=0, label="Diameter (0 = automatic)")
    flow_threshold = mo.ui.number(start=0, stop=3, step=0.05, value=0.8, label="Flow threshold")
    cellprob_threshold = mo.ui.number(
        start=-5, stop=5, step=0.05, value=0.5, label="Cell-probability threshold"
    )
    run_button = mo.ui.run_button(label="Run AC analysis", kind="success")
    return (
        cellprob_threshold,
        data_root,
        diameter,
        flow_threshold,
        model_type,
        run_button,
        segment_high,
        segment_low,
        selected_groups,
    )


@app.cell
def _(
    Path,
    cellprob_threshold,
    data_root,
    diameter,
    flow_threshold,
    mo,
    model_type,
    run_button,
    segment_high,
    segment_low,
    selected_groups,
):
    controls = mo.vstack(
        [
            data_root,
            selected_groups,
            mo.hstack([model_type, diameter]),
            mo.hstack([segment_low, segment_high]),
            mo.hstack([flow_threshold, cellprob_threshold]),
            run_button,
            mo.md("Accelerate selects the runtime device only after **Run AC analysis** is clicked."),
        ]
    )
    settings = {
        "data_root": Path(data_root.value),
        "groups": tuple(selected_groups.value),
        "model_type": model_type.value,
        "segment_low": float(segment_low.value),
        "segment_high": float(segment_high.value),
        "diameter": float(diameter.value),
        "flow_threshold": float(flow_threshold.value),
        "cellprob_threshold": float(cellprob_threshold.value),
    }
    controls
    return (settings,)


@app.cell
def _(run_ac_analysis, run_button, settings):
    analysis = run_ac_analysis(settings) if run_button.value else None
    return (analysis,)


@app.cell
def _(mo):
    selected_fov_name, set_selected_fov_name = mo.state(None, allow_self_loops=True)
    selected_hour, set_selected_hour = mo.state(None, allow_self_loops=True)
    selected_ratio, set_selected_ratio = mo.state(None, allow_self_loops=True)
    return (
        selected_fov_name,
        selected_hour,
        selected_ratio,
        set_selected_fov_name,
        set_selected_hour,
        set_selected_ratio,
    )


@app.cell
def _(
    analysis,
    mo,
    plotting,
    ratios,
    selected_fov_name,
    selected_hour,
    selected_ratio,
    set_selected_fov_name,
    set_selected_hour,
    set_selected_ratio,
):
    mo.stop(
        analysis is None,
        mo.md("Configure the analysis and click **Run AC analysis** to begin segmentation."),
    )
    fov_names = [preview["fov"] for preview in plotting.ordered_previews(analysis["previews"])]
    hour_names = plotting.ordered_analysis_hours(analysis["groups"])
    ratio_names = ratios.RATIO_NAMES

    current_fov_name = selected_fov_name()
    if current_fov_name not in fov_names:
        current_fov_name = fov_names[0] if fov_names else None
        set_selected_fov_name(current_fov_name)

    current_hour = selected_hour()
    if current_hour not in hour_names:
        current_hour = hour_names[0] if hour_names else None
        set_selected_hour(current_hour)

    current_ratio = selected_ratio()
    if current_ratio not in ratio_names:
        current_ratio = ratio_names[0] if ratio_names else None
        set_selected_ratio(current_ratio)

    fov_position = fov_names.index(current_fov_name) if current_fov_name else -1
    hour_position = hour_names.index(current_hour) if current_hour else -1
    ratio_position = ratio_names.index(current_ratio) if current_ratio else -1
    fov_dropdown = mo.ui.dropdown(
        options=fov_names,
        value=current_fov_name,
        label="FOV",
        on_change=set_selected_fov_name,
        disabled=not fov_names,
    )
    previous_fov = mo.ui.button(
        label="←",
        tooltip="Previous FOV",
        disabled=fov_position <= 0,
        on_click=lambda _: set_selected_fov_name(fov_names[fov_position - 1]),
    )
    next_fov = mo.ui.button(
        label="→",
        tooltip="Next FOV",
        disabled=not fov_names or fov_position == len(fov_names) - 1,
        on_click=lambda _: set_selected_fov_name(fov_names[fov_position + 1]),
    )
    hour_dropdown = mo.ui.dropdown(
        options=hour_names,
        value=current_hour,
        label="Hour",
        on_change=set_selected_hour,
        disabled=not hour_names,
    )
    previous_hour = mo.ui.button(
        label="←",
        tooltip="Previous hour",
        disabled=hour_position <= 0,
        on_click=lambda _: set_selected_hour(hour_names[hour_position - 1]),
    )
    next_hour = mo.ui.button(
        label="→",
        tooltip="Next hour",
        disabled=not hour_names or hour_position == len(hour_names) - 1,
        on_click=lambda _: set_selected_hour(hour_names[hour_position + 1]),
    )
    ratio_dropdown = mo.ui.dropdown(
        options=ratio_names,
        value=current_ratio,
        label="Ratio",
        on_change=set_selected_ratio,
        disabled=not ratio_names,
    )
    previous_ratio = mo.ui.button(
        label="←",
        tooltip="Previous ratio",
        disabled=ratio_position <= 0,
        on_click=lambda _: set_selected_ratio(ratio_names[ratio_position - 1]),
    )
    next_ratio = mo.ui.button(
        label="→",
        tooltip="Next ratio",
        disabled=not ratio_names or ratio_position == len(ratio_names) - 1,
        on_click=lambda _: set_selected_ratio(ratio_names[ratio_position + 1]),
    )
    browser_controls = mo.vstack(
        [
            mo.md("## Figure controls"),
            mo.hstack([previous_fov, fov_dropdown, next_fov]),
            mo.hstack([previous_hour, hour_dropdown, next_hour]),
            mo.hstack([previous_ratio, ratio_dropdown, next_ratio]),
        ]
    )
    browser_controls
    return hour_dropdown, ratio_dropdown


@app.cell(hide_code=True)
def _(analysis, mo, plotting, selected_fov_name):
    mo.stop(
        analysis is None,
        mo.md("Configure the analysis and click **Run AC analysis** to begin segmentation."),
    )
    _selected_previews = plotting.ordered_previews(analysis["previews"])
    if not _selected_previews:
        _segmentation_output = mo.md("## Segmentation overlay\n\nNo successful segmentation overlays are available.")
    else:
        _preview_by_name = {preview["fov"]: preview for preview in _selected_previews}
        _selected_fov = selected_fov_name()
        _selected_preview = _preview_by_name.get(_selected_fov, _selected_previews[0])
        _selected_fov_index = _selected_previews.index(_selected_preview)
        _selected_segmentation_figure = plotting.segmentation_figure(_selected_preview)
        _segmentation_output = mo.vstack(
            [
                mo.md(
                    f"## Segmentation overlay\n\n"
                    f"{_selected_preview['fov']} ({_selected_fov_index + 1}/{len(_selected_previews)})"
                ),
                mo.mpl.interactive(_selected_segmentation_figure),
            ]
        )
    _segmentation_output
    return


@app.cell(hide_code=True)
def _(analysis, hour_dropdown, mo, plotting):
    mo.stop(
        analysis is None,
        mo.md("Configure the analysis and click **Run AC analysis** to begin segmentation."),
    )
    _selected_hour = hour_dropdown.value
    _selected_spectra_figure = plotting.stacked_spectra_figure(analysis["groups"], _selected_hour)
    if _selected_spectra_figure is None:
        _spectra_output = mo.md(
            f"## Stacked spectra ({_selected_hour})\n\n"
            "No fitted spectra are available for this time point."
        )
    else:
        _spectra_output = mo.vstack(
            [
                mo.md(f"## Stacked spectra ({_selected_hour})"),
                mo.mpl.interactive(_selected_spectra_figure),
            ]
        )
    _spectra_output
    return


@app.cell(hide_code=True)
def _(analysis, hour_dropdown, mo, plotting, ratio_dropdown):
    mo.stop(
        analysis is None,
        mo.md("Configure the analysis and click **Run AC analysis** to begin segmentation."),
    )
    _selected_hour = hour_dropdown.value
    _ratio_name = ratio_dropdown.value
    _selected_ratio_figure = plotting.ratio_boxplot_figure(
        analysis["ratios"], _ratio_name, _selected_hour
    )
    _selected_ratio_output = (
        mo.md(
            f"## {_ratio_name} ({_selected_hour})\n\n"
            "No finite cell-level ratios are available."
        )
        if _selected_ratio_figure is None
        else mo.vstack(
            [
                mo.md(f"## Condition comparison: {_ratio_name} ({_selected_hour})"),
                mo.mpl.interactive(_selected_ratio_figure),
            ]
        )
    )
    _selected_ratio_output
    return


@app.cell(hide_code=True)
def _(analysis, mo, plotting, ratio_dropdown):
    mo.stop(
        analysis is None,
        mo.md("Configure the analysis and click **Run AC analysis** to begin segmentation."),
    )
    _ratio_name = ratio_dropdown.value
    _trajectory_figure = plotting.ratio_over_time_figure(analysis["ratios"], _ratio_name)
    _trajectory_output = (
        mo.md(
            f"## {_ratio_name} over time\n\n"
            "No finite baseline and treatment ratios are available for a longitudinal comparison."
        )
        if _trajectory_figure is None
        else mo.vstack(
            [
                mo.md(f"## Condition trajectories: {_ratio_name}"),
                mo.mpl.interactive(_trajectory_figure),
            ]
        )
    )
    _trajectory_output
    return


@app.cell
def _(analysis, hour_dropdown, mo, ratio_dropdown):
    mo.stop(
        analysis is None,
        mo.md("Configure the analysis and click **Run AC analysis** to begin segmentation."),
    )
    table_hour = hour_dropdown.value
    table_ratio = ratio_dropdown.value

    def rows_for_hour(frame):
        if frame.empty:
            return frame
        return frame.loc[frame["time"] == table_hour]

    table_peaks = rows_for_hour(analysis["peaks"])
    table_ratios = rows_for_hour(analysis["ratios"])
    table_p_values = rows_for_hour(analysis["p_values"])
    if not table_p_values.empty:
        table_p_values = table_p_values.loc[table_p_values["ratio"] == table_ratio]
    table_peak_count = len(table_peaks)
    table_ratio_count = len(table_ratios)
    mo.vstack(
        [
            mo.md(
                f"## Analysis tables ({table_hour})\n\n"
                f"Runtime device: `{analysis['device']}`"
            ),
            mo.md("### FOV status"),
            analysis["fov_summary"],
            mo.md(
                f"### Cell-level peak fits ({table_hour}) — "
                f"showing {min(table_peak_count, 100)} of {table_peak_count} rows"
            ),
            table_peaks.head(100),
            mo.md(
                f"### Cell-level ratios ({table_hour}) — "
                f"showing {min(table_ratio_count, 100)} of {table_ratio_count} rows"
            ),
            table_ratios.head(100),
            mo.md(f"### Carbon 13 Welch tests: {table_ratio} ({table_hour})"),
            table_p_values,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
