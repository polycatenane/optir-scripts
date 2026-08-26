#!/usr/bin/env python3
"""Segment cells from a provided binary mask and compute ROI statistics.

Inputs
  1) intensity TIFF (expected float32; any numeric dtype is accepted)
  2) mask TIFF (binary; nonzero treated as foreground)

Outputs
  - CSV with ROI measurements (area, perimeter, mean intensity, etc.)
  - Multi-page TIFF containing:
      page 0: original intensity image (float32)
      page 1: label image (uint32; 0=background, 1..N = ROI id)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from skimage import io, measure, segmentation


def write_intensity_and_labels_tiff(out_path: Path, *, intensity: np.ndarray, labels: np.ndarray) -> None:
    """Write a multi-page TIFF: [intensity(float32), labels(uint32)]."""
    try:
        import tifffile  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "tifffile is required to write a multi-page TIFF with mixed dtypes. "
            "Install it with: pip install tifffile"
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    intensity32 = intensity.astype(np.float32, copy=False)
    labels32 = labels.astype(np.uint32, copy=False)
    tifffile.imwrite(str(out_path), [intensity32, labels32])


def _positive_int(value: str) -> int:
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Label a binary mask, filter ROIs by area, write ROI stats CSV, "
            "and save a multi-page TIFF containing the intensity image and the label image."
        )
    )
    p.add_argument("intensity_tiff", help="Input intensity TIFF (ideally float32).")
    p.add_argument("mask_tiff", help="Input mask TIFF (binary; nonzero = foreground).")
    p.add_argument(
        "--min-area",
        type=int,
        default=0,
        help="Minimum ROI area in pixels (inclusive). Default: 0",
    )
    p.add_argument(
        "--max-area",
        type=int,
        default=2**31 - 1,
        help=f"Maximum ROI area in pixels (inclusive). Default: {2**31 - 1}",
    )
    p.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2),
        default=2,
        help="Pixel connectivity for labeling (1=4-neighborhood, 2=8-neighborhood). Default: 2",
    )
    p.add_argument(
        "--csv-out",
        default=None,
        help="CSV output path. Default: <intensity_stem>_rois.csv",
    )
    p.add_argument(
        "--tiff-out",
        default=None,
        help="Output TIFF path. Default: <intensity_stem>_labeled.tif",
    )
    return p.parse_args()


def load_2d_tiff(path: Path, *, name: str) -> np.ndarray:
    arr = io.imread(str(path))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D TIFF; got shape={arr.shape} (ndim={arr.ndim})")
    return np.asarray(arr)


def filter_labels_by_area(lbl: np.ndarray, *, min_area: int, max_area: int) -> np.ndarray:
    if min_area <= 0 and max_area >= (2**31 - 1):
        return lbl

    props = measure.regionprops(lbl)
    keep = {r.label for r in props if (min_area <= r.area <= max_area)}
    if not keep:
        return np.zeros_like(lbl, dtype=np.int32)
    m = np.isin(lbl, list(keep))
    filtered = np.where(m, lbl, 0)
    filtered, _, _ = segmentation.relabel_sequential(filtered)
    return filtered.astype(np.int32, copy=False)


def regionprops_to_csv(
    csv_path: Path, *, lbl: np.ndarray, intensity: np.ndarray
) -> None:
    # Note: some props require 2D.
    props = (
        "label",
        "area",
        "perimeter",
        "centroid",
        "bbox",
        "eccentricity",
        "solidity",
        "equivalent_diameter_area",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "mean_intensity",
        "min_intensity",
        "max_intensity",
    )
    table = measure.regionprops_table(lbl, intensity_image=intensity, properties=props)
    fieldnames = list(table.keys())
    n = 0 if not fieldnames else len(table[fieldnames[0]])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            row = {k: table[k][i] for k in fieldnames}
            w.writerow(row)


def main() -> int:
    args = parse_args()
    intensity_path = Path(args.intensity_tiff)
    mask_path = Path(args.mask_tiff)

    if not intensity_path.exists():
        print(f"Intensity TIFF not found: {intensity_path}", file=sys.stderr)
        return 2
    if not mask_path.exists():
        print(f"Mask TIFF not found: {mask_path}", file=sys.stderr)
        return 2

    intensity = load_2d_tiff(intensity_path, name="intensity_tiff")
    mask_raw = load_2d_tiff(mask_path, name="mask_tiff")
    if mask_raw.shape != intensity.shape:
        print(
            f"Shape mismatch: intensity={intensity.shape} mask={mask_raw.shape}",
            file=sys.stderr,
        )
        return 2

    mask = mask_raw.astype(np.float32, copy=False) > 0.0
    lbl0 = measure.label(mask, connectivity=args.connectivity)
    lbl = filter_labels_by_area(lbl0, min_area=args.min_area, max_area=args.max_area)

    stem = intensity_path.with_suffix("").name
    csv_out = Path(args.csv_out) if args.csv_out else intensity_path.with_name(f"{stem}_rois.csv")
    tiff_out = Path(args.tiff_out) if args.tiff_out else intensity_path.with_name(f"{stem}_labeled.tif")

    regionprops_to_csv(csv_out, lbl=lbl, intensity=intensity.astype(np.float32, copy=False))

    write_intensity_and_labels_tiff(
        tiff_out,
        intensity=intensity,
        labels=lbl,
    )

    n_rois = int(lbl.max())
    print(f"ROIs: {n_rois}")
    print(f"Wrote CSV: {csv_out}")
    print(f"Wrote labeled TIFF: {tiff_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
