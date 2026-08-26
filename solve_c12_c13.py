#!/usr/bin/env python3
"""Solve for c12 and c13 from a 2x2 linear system.

Equations (constants h12, h13, i12, i13):

    X = c12*h12 + c13*h13
    Y = c12*i12 + c13*i13

where,

X: treatment 1648cm-1 intensity
Y: treatment 1612cm-1 intensity

h12 : control 1648cm-1 intensity
i12 : control 1612cm-1 intensity
h13 : c13 1648cm-1 intensity
i13 : c13 1612cm-1 intensity

Modes:

1) Scalar: provide X and Y on the command line.
2) Per-ROI: provide two ROI CSVs written by [`segment_from_mask.py`](segment_from_mask.py:1).
   Each CSV provides a per-ROI `mean_intensity` column. One CSV defines X, the other defines Y.
   The script solves (c12,c13) for each ROI and writes an output CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _read_roi_mean_intensity_csv(path: Path) -> dict[int, float]:
    """Read ROI stats CSV from [`segment_from_mask.py`](segment_from_mask.py:111).

    Requires columns: label, mean_intensity
    Returns: {label: mean_intensity}
    """

    import csv

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        if "label" not in r.fieldnames or "mean_intensity" not in r.fieldnames:
            raise ValueError(
                f"CSV must include columns 'label' and 'mean_intensity'; got {r.fieldnames} in {path}"
            )

        out: dict[int, float] = {}
        for row in r:
            try:
                lbl = int(float(row["label"]))
                mean_intensity = float(row["mean_intensity"])  # already ROI mean
            except Exception as e:
                raise ValueError(f"Failed parsing row in {path}: {row}") from e
            out[lbl] = mean_intensity
    return out


def _find_unique_csv(
    csv_dir: Path,
    *,
    must_contain: tuple[str, ...],
    recursive: bool,
) -> Path:
    tokens = tuple(t.lower() for t in must_contain)

    if recursive:
        candidates = [p for p in csv_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"]
    else:
        candidates = [p for p in csv_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]

    def ok(p: Path) -> bool:
        name = p.name.lower()
        return all(tok in name for tok in tokens)

    matches = [p for p in candidates if ok(p)]
    if len(matches) == 0:
        raise FileNotFoundError(
            "No matching ROI CSV found. Need filename containing: "
            + ", ".join(repr(t) for t in must_contain)
            + f" (dir={csv_dir})"
        )
    if len(matches) > 1:
        msg = "\n".join(str(p) for p in sorted(matches))
        raise FileExistsError(
            "Multiple matching ROI CSVs found; refine tokens or make directory unique.\n"
            f"Tokens: {must_contain}\nMatches:\n{msg}"
        )
    return matches[0]

def solve_c12_c13(
    X: float,
    Y: float,
    *,
    h12: float,
    h13: float,
    i12: float,
    i13: float,
) -> tuple[float, float]:
    """Return (c12, c13) solving:

    X = c12*h12 + c13*h13
    Y = c12*i12 + c13*i13
    """

    A = np.array([[h12, h13], [i12, i13]], dtype=np.float64)
    b = np.array([X, Y], dtype=np.float64)
    try:
        c12, c13 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        det = float(h12 * i13 - h13 * i12)
        raise ValueError(
            "Singular system: cannot solve uniquely. "
            f"det(h12,h13;i12,i13)={det} (h12={h12}, h13={h13}, i12={i12}, i13={i13})"
        ) from e
    return float(c12), float(c13)

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Solve X=c12*h12+c13*h13 and Y=c12*i12+c13*i13 for c12,c13. "
            "Provide X and Y directly, or pass ROI CSVs from segment_from_mask.py to solve per ROI."
        )
    )
    p.add_argument("X", type=float, nargs="?", help="Right-hand-side value for the first equation")
    p.add_argument("Y", type=float, nargs="?", help="Right-hand-side value for the second equation")

    p.add_argument(
        "--roi-csv-x",
        type=Path,
        default=None,
        help="ROI stats CSV whose mean_intensity defines X (output of segment_from_mask.py)",
    )
    p.add_argument(
        "--roi-csv-y",
        type=Path,
        default=None,
        help="ROI stats CSV whose mean_intensity defines Y (output of segment_from_mask.py)",
    )
    p.add_argument(
        "--roi-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing ROI CSVs. If set, --roi-csv-x/--roi-csv-y are auto-selected by keywords in the "
            "CSV filename (defaults: X token=1648, Y token=1612)."
        ),
    )
    p.add_argument(
        "--x-token",
        type=str,
        default="1648",
        help="Keyword to identify the X CSV by filename (default: 1648)",
    )
    p.add_argument(
        "--y-token",
        type=str,
        default="1612",
        help="Keyword to identify the Y CSV by filename (default: 1612)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Search --roi-dir recursively",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV for per-ROI results. Default: saved into --roi-csv-dir (if provided), otherwise next to "
            "--roi-csv-x."
        ),
    )

    p.add_argument("--h12", type=float, default=19.18179715
, help="Constant h12 (default: 2.772)")
    p.add_argument("--h13", type=float, default=14.50560917
, help="Constant h13 (default: 4.478)")
    p.add_argument("--i12", type=float, default=8.334610562
, help="Constant i12 (default: 1.175)")
    p.add_argument("--i13", type=float, default=28.72629824
, help="Constant i13 (default: 10.865)")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # Per-ROI mode
    roi_csv_x: Path | None = args.roi_csv_x
    roi_csv_y: Path | None = args.roi_csv_y
    if args.roi_dir is not None:
        csv_dir: Path = args.roi_dir
        if not csv_dir.exists() or not csv_dir.is_dir():
            raise FileNotFoundError(f"--roi-dir must be an existing directory: {csv_dir}")
        if roi_csv_x is None:
            roi_csv_x = _find_unique_csv(csv_dir, must_contain=(args.x_token,), recursive=args.recursive)
        if roi_csv_y is None:
            roi_csv_y = _find_unique_csv(csv_dir, must_contain=(args.y_token,), recursive=args.recursive)

    if roi_csv_x is not None or roi_csv_y is not None:
        if args.roi_csv_x is None or args.roi_csv_y is None:
            if roi_csv_x is None or roi_csv_y is None:
                print(
                    "Provide both --roi-csv-x and --roi-csv-y, or pass --roi-dir for auto-discovery.",
                    file=sys.stderr,
                )
                return 2
        if not roi_csv_x.exists():
            raise FileNotFoundError(f"ROI CSV X not found: {roi_csv_x}")
        if not roi_csv_y.exists():
            raise FileNotFoundError(f"ROI CSV Y not found: {roi_csv_y}")

        x_by_label = _read_roi_mean_intensity_csv(roi_csv_x)
        y_by_label = _read_roi_mean_intensity_csv(roi_csv_y)
        labels = sorted(set(x_by_label).intersection(y_by_label))
        if not labels:
            raise ValueError(
                "No common ROI labels between the two CSVs. "
                "Both CSVs must come from the same ROI labeling so ROI ids match."
            )

        Xs = np.array([x_by_label[l] for l in labels], dtype=np.float64)
        Ys = np.array([y_by_label[l] for l in labels], dtype=np.float64)
        c12s = np.empty_like(Xs)
        c13s = np.empty_like(Xs)

        for idx, (X, Y) in enumerate(zip(Xs, Ys, strict=True)):
            c12, c13 = solve_c12_c13(
                float(X),
                float(Y),
                h12=float(args.h12),
                h13=float(args.h13),
                i12=float(args.i12),
                i13=float(args.i13),
            )
            c12s[idx] = c12
            c13s[idx] = c13

        if args.out_csv is not None:
            out_csv: Path = args.out_csv
        elif args.roi_dir is not None:
            out_csv = Path(args.roi_dir) / "c12_c13_rois.csv"
        else:
            out_csv = roi_csv_x.parent / "c12_c13_rois.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        import csv

        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["label", "X", "Y", "c12", "c13"])
            w.writeheader()
            for l, X, Y, c12, c13 in zip(labels, Xs, Ys, c12s, c13s, strict=True):
                w.writerow({"label": l, "X": float(X), "Y": float(Y), "c12": float(c12), "c13": float(c13)})

        print(f"ROIs (common labels): {len(labels)}")
        print(f"Read X ROI CSV: {roi_csv_x}")
        print(f"Read Y ROI CSV: {roi_csv_y}")
        print(f"Wrote per-ROI results CSV: {out_csv}")
        return 0

    # Scalar mode
    if args.X is None or args.Y is None:
        print("Provide X and Y, or pass --roi-csv-x/--roi-csv-y for per-ROI solving.", file=sys.stderr)
        return 2
    X = args.X
    Y = args.Y

    c12, c13 = solve_c12_c13(
        X,
        Y,
        h12=args.h12,
        h13=args.h13,
        i12=args.i12,
        i13=args.i13,
    )
    print(f"c12 = {c12}")
    print(f"c13 = {c13}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
