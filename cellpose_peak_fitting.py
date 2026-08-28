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


@app.cell
def _():
    from collections import defaultdict
    from functools import reduce
    from pathlib import Path
    import re

    from accelerate import Accelerator
    from cellpose import models
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.optimize import least_squares

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
    FORCED_PEAK_CENTERS = (1250.0, 1375.0, 1530.0, 1550.0, 1612.0, 1656.0)
    FORCED_PEAK_TOLERANCE = 3.0
    MIN_FWHM = 40.0
    MAX_FWHM = 80.0
    MIN_SIGMA = MIN_FWHM / 2.354820045
    MAX_SIGMA = MAX_FWHM / 2.354820045
    return (
        Accelerator,
        DATA_ROOT,
        DEFAULT_GROUPS,
        FORCED_PEAK_CENTERS,
        FORCED_PEAK_TOLERANCE,
        MAX_SIGMA,
        MIN_SIGMA,
        defaultdict,
        least_squares,
        mo,
        models,
        np,
        pd,
        plt,
        re,
        reduce,
        stats,
    )


@app.cell
def _(np, pd, re):
    def parse_wavenumber(path):
        match = re.search(r"_(\d+(?:\.\d+)?)cm-1_", path.name)
        if match is None:
            raise ValueError(f"Cannot read wavenumber from {path.name}")
        return float(match.group(1))


    def parse_date(path):
        match = re.search(r"(?:^|_)(\d{8})(?:_|$)", path.stem)
        if match is None:
            raise ValueError(f"Cannot read acquisition date from {path.name}")
        return match.group(1)


    def group_details(root, fov_dir):
        parts = list(fov_dir.relative_to(root).parts)
        if parts and re.fullmatch(r"FOV\d+", parts[-1], flags=re.IGNORECASE):
            parts.pop()
        if not parts:
            raise ValueError(f"Cannot derive a group from {fov_dir}")
        return {
            "group": " / ".join(parts),
            "condition": parts[0],
            "time": parts[1] if len(parts) > 1 else None,
        }


    def scan_ac_fovs(root, groups):
        records = []
        for group_name in groups:
            group_dir = root / group_name
            if not group_dir.is_dir():
                records.append(
                    {
                        "status": "missing_group",
                        "fov_dir": group_dir,
                        "message": f"Selected group does not exist: {group_dir}",
                    }
                )
                continue
            files_by_fov = {}
            for path in group_dir.rglob("*_AC.csv"):
                files_by_fov.setdefault(path.parent, []).append(path)
            for fov_dir, paths in sorted(files_by_fov.items(), key=lambda item: str(item[0])):
                details = group_details(root, fov_dir)
                records.append(
                    {
                        "status": "pending",
                        "fov_dir": fov_dir,
                        "relative_fov": str(fov_dir.relative_to(root)),
                        "files": sorted(paths, key=parse_wavenumber),
                        **details,
                    }
                )
        return records


    def read_ac_cube(paths):
        planes = []
        wavelengths = []
        expected_shape = None
        seen_wavenumbers = set()
        for path in paths:
            wavenumber = parse_wavenumber(path)
            if wavenumber in seen_wavenumbers:
                raise ValueError(f"Duplicate AC plane at {wavenumber:g} cm-1: {path.parent}")
            seen_wavenumbers.add(wavenumber)
            plane = np.loadtxt(path, delimiter=",", dtype=np.float32)
            if plane.ndim != 2:
                raise ValueError(f"AC plane is not two-dimensional: {path}")
            if expected_shape is None:
                expected_shape = plane.shape
            elif plane.shape != expected_shape:
                raise ValueError(
                    f"Inconsistent AC plane shape in {path.parent}: "
                    f"expected {expected_shape}, received {plane.shape}"
                )
            wavelengths.append(wavenumber)
            planes.append(plane)
        if not planes:
            raise ValueError("No AC planes found")
        order = np.argsort(wavelengths)
        x = np.asarray(wavelengths, dtype=float)[order]
        cube = np.stack([planes[index] for index in order], axis=-1)
        return cube, x


    def read_ir_profiles(root):
        profile_dir = root / "Power data Processed"
        profiles = {}
        for path in profile_dir.glob("*.csv"):
            date = parse_date(path)
            frame = pd.read_csv(path)
            required = {"Wavenumber_cm-1", "Averaged_Power_mV"}
            if not required.issubset(frame.columns):
                raise ValueError(f"Unexpected IR-power columns in {path.name}")
            profile = frame.loc[:, ["Wavenumber_cm-1", "Averaged_Power_mV"]].dropna()
            profile = profile.sort_values("Wavenumber_cm-1")
            profiles[date] = (
                profile["Wavenumber_cm-1"].to_numpy(dtype=float),
                profile["Averaged_Power_mV"].to_numpy(dtype=float),
            )
        if not profiles:
            raise FileNotFoundError(f"No processed IR-power CSV files in {profile_dir}")
        return profiles


    def align_to_ir_power(x, raw_spectra, date, profiles):
        if date not in profiles:
            raise ValueError(f"No processed IR-power profile matches acquisition date {date}")
        reference_x, reference_power = profiles[date]
        supported = (x >= reference_x.min()) & (x <= reference_x.max())
        if supported.sum() < 3:
            raise ValueError(f"Too little overlap between AC data and IR-power profile for {date}")
        x_supported = x[supported]
        power = np.interp(x_supported, reference_x, reference_power)
        valid = np.isfinite(power) & (power > 0)
        if valid.sum() < 3:
            raise ValueError(f"IR-power profile has insufficient positive values for {date}")
        return x_supported[valid], raw_spectra[:, supported][:, valid] / power[valid]


    def extract_label_spectra(ac_cube, masks):
        if masks.shape != ac_cube.shape[:2]:
            raise ValueError(
                f"Mask/AC shape mismatch: mask={masks.shape}, AC image={ac_cube.shape[:2]}"
            )
        labels = np.unique(masks)
        labels = labels[labels > 0]
        spectra = []
        pixel_counts = []
        kept_labels = []
        for label in labels:
            enclosed_ac = ac_cube[masks == label, :]
            if enclosed_ac.size == 0:
                continue
            spectra.append(np.nanmean(enclosed_ac, axis=0))
            pixel_counts.append(int(enclosed_ac.shape[0]))
            kept_labels.append(int(label))
        if not spectra:
            return np.empty((0, ac_cube.shape[-1]), dtype=float), np.array([], dtype=int), np.array([], dtype=int)
        return np.vstack(spectra), np.asarray(kept_labels), np.asarray(pixel_counts)


    return (
        align_to_ir_power,
        extract_label_spectra,
        read_ac_cube,
        read_ir_profiles,
        scan_ac_fovs,
    )


@app.cell
def _(
    FORCED_PEAK_CENTERS,
    FORCED_PEAK_TOLERANCE,
    MAX_SIGMA,
    MIN_SIGMA,
    least_squares,
    np,
    pd,
):
    def gaussian(x, amplitude, center, sigma):
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


    def estimate_polynomial_baseline(x, y, low_fraction=0.25):
        threshold = np.nanquantile(y, low_fraction)
        low = y <= threshold
        if np.count_nonzero(low) < 3:
            low = np.ones_like(y, dtype=bool)
        x_fit = x[low]
        y_fit = y[low]
        x_mid = float(np.mean(x_fit))
        y_mid = float(np.mean(y_fit))
        result = least_squares(
            lambda p: p[0] * (x_fit - p[1]) ** 2 + p[2] - y_fit,
            x0=np.array([-1e-6, x_mid, y_mid]),
            bounds=([-np.inf, float(x.min()), -np.inf], [0.0, float(x.max()), np.inf]),
            max_nfev=5000,
        )
        a, h, k = result.x
        return a * (x - h) ** 2 + k


    def fit_single_spectrum(x, y):
        baseline = estimate_polynomial_baseline(x, y)
        corrected = y - baseline
        centers = [
            center
            for center in FORCED_PEAK_CENTERS
            if x.min() <= center <= x.max()
        ]
        if not centers:
            return {
                "fit": baseline,
                "baseline": baseline,
                "components": np.empty((0, x.size)),
                "peaks": pd.DataFrame(),
                "r_squared": np.nan,
                "rmse": np.nan,
            }
        signal_span = max(float(np.nanmax(corrected) - np.nanmin(corrected)), 1e-6)
        initial = []
        lower = []
        upper = []
        dx = float(np.median(np.diff(x)))
        sigma_seed = float(np.clip(1.5 * abs(dx), MIN_SIGMA, MAX_SIGMA))
        for center in centers:
            nearby = np.where(np.abs(x - center) <= FORCED_PEAK_TOLERANCE)[0]
            index = int(nearby[np.argmax(corrected[nearby])]) if nearby.size else int(np.argmin(np.abs(x - center)))
            amplitude = max(float(corrected[index] - np.nanpercentile(corrected, 5)), 1e-6)
            initial.extend([amplitude, float(x[index]), sigma_seed])
            lower.extend([0.0, center - FORCED_PEAK_TOLERANCE, MIN_SIGMA])
            upper.extend([1.5 * signal_span + abs(float(np.nanmax(corrected))), center + FORCED_PEAK_TOLERANCE, MAX_SIGMA])

        def model(parameters):
            components = np.asarray(
                [gaussian(x, *row) for row in parameters.reshape(-1, 3)], dtype=float
            )
            return components.sum(axis=0), components

        result = least_squares(
            lambda parameters: model(parameters)[0] - corrected,
            x0=np.asarray(initial),
            bounds=(np.asarray(lower), np.asarray(upper)),
            loss="soft_l1",
            f_scale=max(float(np.nanstd(corrected)) * 0.05, 1e-3),
            max_nfev=10000,
        )
        fitted_corrected, components = model(result.x)
        fit = baseline + fitted_corrected
        residual = y - fit
        total = float(np.sum((y - np.nanmean(y)) ** 2))
        peaks = []
        for index, (amplitude, center, sigma) in enumerate(result.x.reshape(-1, 3)):
            component = components[index]
            peaks.append(
                {
                    "forced_center_cm-1": centers[index],
                    "center_cm-1": float(center),
                    "peak_height": float(np.max(component)),
                    "amplitude": float(amplitude),
                    "sigma_cm-1": float(sigma),
                    "fwhm_cm-1": float(2.354820045 * sigma),
                    "area": float(amplitude * sigma * np.sqrt(2 * np.pi)),
                }
            )
        return {
            "fit": fit,
            "baseline": baseline,
            "components": components,
            "peaks": pd.DataFrame(peaks),
            "r_squared": 1.0 if total == 0 else 1.0 - float(np.sum(residual**2)) / total,
            "rmse": float(np.sqrt(np.mean(residual**2))),
        }


    def fit_group_spectra(x, spectra, metadata):
        fit_rows = []
        peak_tables = []
        fit_matrix = np.empty_like(spectra, dtype=float)
        baseline_matrix = np.empty_like(spectra, dtype=float)
        results = []
        for index, spectrum in enumerate(spectra):
            result = fit_single_spectrum(x, spectrum)
            results.append(result)
            fit_matrix[index] = result["fit"]
            baseline_matrix[index] = result["baseline"]
            row = metadata.iloc[index].to_dict()
            fit_rows.append({**row, "r_squared": result["r_squared"], "rmse": result["rmse"]})
            if not result["peaks"].empty:
                peak_tables.append(result["peaks"].assign(**row))
        return {
            "fit_results": results,
            "fit_matrix": fit_matrix,
            "baseline_matrix": baseline_matrix,
            "fit_summary": pd.DataFrame(fit_rows),
            "peaks": pd.concat(peak_tables, ignore_index=True) if peak_tables else pd.DataFrame(),
        }


    return (fit_group_spectra,)


@app.cell
def _(
    Accelerator,
    align_to_ir_power,
    defaultdict,
    extract_label_spectra,
    fit_group_spectra,
    models,
    np,
    pd,
    read_ac_cube,
    read_ir_profiles,
    reduce,
    scan_ac_fovs,
    stats,
):
    def run_ac_analysis(settings):
        root = settings["data_root"]
        if not root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {root}")
        profiles = read_ir_profiles(root)
        accelerator = Accelerator()
        device = accelerator.device
        model = models.CellposeModel(
            gpu=device.type != "cpu",
            device=device,
            model_type=settings["model_type"],
        )
        records = scan_ac_fovs(root, settings["groups"])
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
                ac_cube, x = read_ac_cube(record["files"])
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
                raw_spectra, labels, pixel_counts = extract_label_spectra(ac_cube, masks)
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
                x_supported, corrected = align_to_ir_power(x, raw_spectra, date, profiles)
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
            fitted = fit_group_spectra(shared_x, normalized, metadata)
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
        ratios, p_values = summarize_ratios(peaks)
        return {
            "device": str(device),
            "fov_summary": pd.DataFrame(fov_summary),
            "groups": groups,
            "peaks": peaks,
            "fit_summary": fit_summary,
            "ratios": ratios,
            "p_values": p_values,
            "previews": previews,
        }


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

        wide["1612/(1530+1550)"] = peak(1612.0) / (peak(1530.0) + peak(1550.0))
        wide["1530/(1530+1550)"] = peak(1530.0) / (peak(1530.0) + peak(1550.0))
        wide["1612/(1612+1656)"] = peak(1612.0) / (peak(1612.0) + peak(1656.0))
        ratio_columns = ["1612/(1530+1550)", "1530/(1530+1550)", "1612/(1612+1656)"]
        rows = []
        for (group, condition, time), frame in wide.groupby(["group", "condition", "time"], dropna=False):
            for ratio in ratio_columns:
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
                for ratio in ratio_columns:
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
            mo.md("# AC spectral segmentation and Gaussian peak fitting"),
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
        "data_root": __import__("pathlib").Path(data_root.value),
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
def _(DEFAULT_GROUPS, np, plt, re):
    RATIO_COLUMNS = (
        "1612/(1530+1550)",
        "1530/(1530+1550)",
        "1612/(1612+1656)",
    )

    def group_sort_key(condition, time, group):
        if condition == "0 hour":
            return (0, 0, 0, group)
        match = re.search(r"(\d+)", str(time))
        time_order = int(match.group(1)) if match else 999
        if condition == "Carbon 13":
            condition_order = 0
        else:
            condition_order = DEFAULT_GROUPS.index(condition) + 1 if condition in DEFAULT_GROUPS else 999
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


    def group_fit_figure(group_name, group):
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(
            group["x"],
            np.nanmean(group["spectra"], axis=0),
            color="black",
            label="Mean normalized AC",
        )
        axis.plot(
            group["x"],
            np.nanmean(group["fit_matrix"], axis=0),
            color="crimson",
            label="Mean Gaussian fit",
        )
        axis.set_title(group_name)
        axis.set_xlabel("Wavenumber (cm-1)")
        axis.set_ylabel("Normalized intensity")
        axis.legend()
        figure.tight_layout()
        return figure


    def ratio_boxplot_figure(ratios):
        if ratios.empty:
            return None
        group_names = ordered_group_names(ratios)
        figure, axes = plt.subplots(
            nrows=len(RATIO_COLUMNS),
            ncols=1,
            figsize=(max(9, 1.65 * len(group_names)), 4 * len(RATIO_COLUMNS)),
            squeeze=False,
        )
        for axis, ratio_name in zip(axes[:, 0], RATIO_COLUMNS, strict=True):
            series = []
            labels = []
            for group_name in group_names:
                values = (
                    ratios.loc[ratios["group"] == group_name, ratio_name]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy(dtype=float)
                )
                if values.size:
                    series.append(values)
                    labels.append(group_name)
            if not series:
                axis.set_title(f"{ratio_name} (no finite cell values)")
                axis.axis("off")
                continue
            positions = np.arange(1, len(series) + 1)
            axis.boxplot(series, positions=positions, showfliers=False, patch_artist=True)
            for position, values in zip(positions, series, strict=True):
                jitter = (
                    np.array([float(position)])
                    if values.size == 1
                    else position + np.linspace(-0.18, 0.18, num=values.size)
                )
                axis.scatter(jitter, values, s=12, alpha=0.55, linewidths=0, zorder=3)
            axis.set_title(ratio_name)
            axis.set_ylabel("Cell-level ratio")
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.grid(True, axis="y", alpha=0.25)
            if ratio_name != "1612/(1530+1550)":
                axis.set_ylim(0, 1)
        figure.tight_layout()
        return figure


    return (
        group_fit_figure,
        ordered_analysis_groups,
        ordered_group_names,
        ordered_previews,
        ratio_boxplot_figure,
        segmentation_figure,
    )


@app.cell
def _(
    analysis,
    mo,
    ordered_analysis_groups,
    ordered_previews,
):
    mo.stop(
        analysis is None,
        mo.md("Configure the analysis and click **Run AC analysis** to begin segmentation."),
    )
    preview_count = len(ordered_previews(analysis["previews"]))
    group_count = len(ordered_analysis_groups(analysis["groups"]))
    view_mode = mo.ui.dropdown(
        options=["Segmentation overlay", "Group fit", "Ratio distributions"],
        value="Segmentation overlay",
        label="Figure view",
    )
    fov_index = mo.ui.slider(
        start=0,
        stop=max(preview_count - 1, 0),
        step=1,
        value=0,
        label=f"FOV index (0–{max(preview_count - 1, 0)})",
    )
    group_index = mo.ui.slider(
        start=0,
        stop=max(group_count - 1, 0),
        step=1,
        value=0,
        label=f"Group index (0–{max(group_count - 1, 0)})",
    )
    browser_controls = mo.vstack(
        [
            mo.md("## Figure browser"),
            view_mode,
            fov_index,
            group_index,
            mo.md("The FOV slider applies to segmentation overlays; the group slider applies to group fits."),
        ]
    )
    browser_controls
    return fov_index, group_index, view_mode


@app.cell
def _(
    analysis,
    fov_index,
    group_fit_figure,
    group_index,
    mo,
    ordered_analysis_groups,
    ordered_previews,
    ratio_boxplot_figure,
    segmentation_figure,
    view_mode,
):
    mo.stop(
        analysis is None,
        mo.md("Configure the analysis and click **Run AC analysis** to begin segmentation."),
    )
    if view_mode.value == "Segmentation overlay":
        previews = ordered_previews(analysis["previews"])
        if previews:
            index = min(max(int(fov_index.value), 0), len(previews) - 1)
            figure = segmentation_figure(previews[index])
            figure_label = f"Segmentation overlay: {previews[index]['fov']} ({index + 1}/{len(previews)})"
        else:
            figure = mo.md("No successful segmentation overlays are available.")
            figure_label = "Segmentation overlay"
    elif view_mode.value == "Group fit":
        group_names = ordered_analysis_groups(analysis["groups"])
        if group_names:
            index = min(max(int(group_index.value), 0), len(group_names) - 1)
            group_name = group_names[index]
            figure = group_fit_figure(group_name, analysis["groups"][group_name])
            figure_label = f"Group fit: {group_name} ({index + 1}/{len(group_names)})"
        else:
            figure = mo.md("No group fits are available.")
            figure_label = "Group fit"
    else:
        figure = ratio_boxplot_figure(analysis["ratios"])
        if figure is None:
            figure = mo.md("No finite cell-level ratios are available.")
        figure_label = "Cell-level ratio distributions"

    peak_count = len(analysis["peaks"])
    ratio_count = len(analysis["ratios"])
    mo.vstack(
        [
            mo.md(f"## Analysis device: `{analysis['device']}`"),
            mo.md(f"### {figure_label}"),
            figure,
            mo.md("### FOV status"),
            analysis["fov_summary"],
            mo.md(f"### Cell-level peak fits — showing {min(peak_count, 100)} of {peak_count} rows"),
            analysis["peaks"].head(100),
            mo.md(f"### Cell-level ratios — showing {min(ratio_count, 100)} of {ratio_count} rows"),
            analysis["ratios"].head(100),
            mo.md("### Carbon 13 Welch tests"),
            analysis["p_values"],
        ]
    )


if __name__ == "__main__":
    app.run()
