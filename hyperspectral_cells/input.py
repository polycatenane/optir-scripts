import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_wavenumber(path: Path) -> float:
    match = re.search(r"_(\d+(?:\.\d+)?)cm-1_", path.name)
    if match is None:
        raise ValueError(f"Cannot read wavenumber from {path.name}")
    return float(match.group(1))


def parse_date(path: Path) -> str:
    match = re.search(r"(?:^|_)(\d{8})(?:_|$)", path.stem)
    if match is None:
        raise ValueError(f"Cannot read acquisition date from {path.name}")
    return match.group(1)


def group_details(root: Path, fov_dir: Path) -> dict[str, str | None]:
    parts = list(fov_dir.relative_to(root).parts)
    if parts and re.fullmatch(r"FOV\d+", parts[-1], flags=re.IGNORECASE):
        parts.pop()
    if not parts:
        raise ValueError(f"Cannot derive a group from {fov_dir}")
    condition = parts[0]
    return {
        "group": " / ".join(parts),
        "condition": condition,
        "time": "0 hour" if condition == "0 hour" else parts[1] if len(parts) > 1 else None,
    }


def scan_ac_fovs(root: Path, groups: tuple[str, ...]) -> list[dict]:
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


def read_ac_cube(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
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


def read_ir_profiles(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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


def align_to_ir_power(
    x: np.ndarray,
    raw_spectra: np.ndarray,
    date: str,
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
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


def extract_label_spectra(
    ac_cube: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return (
            np.empty((0, ac_cube.shape[-1]), dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )
    return np.vstack(spectra), np.asarray(kept_labels), np.asarray(pixel_counts)
