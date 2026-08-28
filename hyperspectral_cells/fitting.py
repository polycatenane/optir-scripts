import numpy as np
import pandas as pd
from scipy.optimize import least_squares


FORCED_PEAK_CENTERS = (1250.0, 1375.0, 1530.0, 1550.0, 1612.0, 1656.0)
FORCED_PEAK_TOLERANCE = 3.0
MIN_FWHM = 40.0
MAX_FWHM = 80.0
MIN_SIGMA = MIN_FWHM / 2.354820045
MAX_SIGMA = MAX_FWHM / 2.354820045


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
    centers = [center for center in FORCED_PEAK_CENTERS if x.min() <= center <= x.max()]
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
