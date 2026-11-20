from pathlib import Path

import numpy as np
import pandas as pd

from hyperspectral_cells.input import parse_date, parse_wavenumber

POWER_COLUMNS = ["Wavenumber_cm-1", "Averaged_Power_mV"]


def read_ac_power(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*_AC.csv"))
    if not paths:
        raise FileNotFoundError(f"No AC power CSV files in {directory}")
    dates = {parse_date(path) for path in paths}
    if len(dates) != 1:
        raise ValueError(f"Expected one acquisition date in {directory}, found {sorted(dates)}")
    rows = []
    for path in paths:
        values = np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)
        rows.append(
            {
                "Wavenumber_cm-1": parse_wavenumber(path),
                "Averaged_Power_mV": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "pixel_count": int(values.size),
                "source_file": path.name,
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values("Wavenumber_cm-1", ignore_index=True)
    duplicates = diagnostics.loc[diagnostics["Wavenumber_cm-1"].duplicated(), "Wavenumber_cm-1"]
    if not duplicates.empty:
        raise ValueError(f"Duplicate AC power wavenumbers: {duplicates.tolist()}")
    diagnostics.attrs["date"] = dates.pop()
    return diagnostics


def power_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    return diagnostics.loc[:, POWER_COLUMNS]


def low_pass_fft(diagnostics: pd.DataFrame, shortest_period_cm: float) -> pd.DataFrame:
    x = diagnostics["Wavenumber_cm-1"].to_numpy(float)
    y = diagnostics["Averaged_Power_mV"].to_numpy(float)
    spacing = np.diff(x)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("FFT filtering requires evenly spaced wavenumbers")
    trend = np.linspace(y[0], y[-1], y.size)
    spectrum = np.fft.rfft(y - trend)
    frequencies = np.fft.rfftfreq(y.size, d=spacing[0])
    spectrum[frequencies > 1 / shortest_period_cm] = 0
    filtered = diagnostics.assign(
        raw_mV=y,
        Averaged_Power_mV=np.fft.irfft(spectrum, n=y.size) + trend,
    )
    filtered.attrs = diagnostics.attrs
    return filtered


def power_output_path(output_dir: Path, date: str) -> Path:
    return output_dir / f"power_{date}_processed.csv"


def write_power(diagnostics: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = power_output_path(output_dir, diagnostics.attrs["date"])
    power_table(diagnostics).to_csv(path, index=False)
    return path
