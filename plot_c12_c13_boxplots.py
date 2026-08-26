#!/usr/bin/env python3
"""Box-plot per-ROI solve outputs from [`solve_c12_c13.py`](solve_c12_c13.py:1).

Reads one or more CSVs written by solve_c12_c13.py (per-ROI mode) and produces
box plots per series.

Default metric: per-ROI ratio `c13 / (c12 + c13)`.

Each input CSV is treated as one *series* (e.g., one sample/condition).

Examples:

  # Show interactive window with ratio boxplot for one series
  ./plot_c12_c13_boxplots.py ./c12_c13_rois.csv

  # Compare multiple series (series names inferred from filenames)
  ./plot_c12_c13_boxplots.py ./sampleA.csv ./sampleB.csv

  # Explicit series names and save to PNG
  ./plot_c12_c13_boxplots.py ./a.csv ./b.csv --names A,B --out ./my_boxplots.png

  # Plot other columns too
  ./plot_c12_c13_boxplots.py ./a.csv ./b.csv --columns ratio,c12,c13,X,Y
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_COLUMNS: tuple[str, ...] = ("ratio",)


def _parse_columns(s: str) -> list[str]:
    cols = [c.strip() for c in s.split(",") if c.strip()]
    if not cols:
        raise ValueError("--columns must contain at least one column name")
    return cols


def _parse_names(s: str) -> list[str]:
    names = [n.strip() for n in s.split(",") if n.strip()]
    if not names:
        raise ValueError("--names must contain at least one name")
    return names


def _read_numeric_columns(csv_path: Path, *, columns: list[str]) -> dict[str, list[float]]:
    """Return {column: [values...]} from a CSV.

    Special column name:
      - ratio: computed as c13/(c12+c13) per row (requires c12 and c13)
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out: dict[str, list[float]] = {c: [] for c in columns}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        requires_ratio = "ratio" in columns
        missing = [c for c in columns if c != "ratio" and c not in r.fieldnames]
        if missing:
            raise ValueError(
                f"CSV is missing requested columns {missing}; available columns: {r.fieldnames} (file={csv_path})"
            )
        if requires_ratio and ("c12" not in r.fieldnames or "c13" not in r.fieldnames):
            raise ValueError(
                f"CSV must contain columns 'c12' and 'c13' to compute ratio; available columns: {r.fieldnames} "
                f"(file={csv_path})"
            )

        for row in r:
            # Precompute ratio once per row if requested.
            ratio_val: float | None = None
            if requires_ratio:
                c12_raw = row.get("c12", "")
                c13_raw = row.get("c13", "")
                if c12_raw is not None and c13_raw is not None and str(c12_raw).strip() != "" and str(c13_raw).strip() != "":
                    try:
                        c12 = float(c12_raw)
                        c13 = float(c13_raw)
                    except Exception as e:
                        raise ValueError(
                            f"Failed parsing float in {csv_path} columns 'c12'/'c13': c12={c12_raw!r}, c13={c13_raw!r}"
                        ) from e
                    denom = c12 + c13
                    if denom != 0:
                        ratio_val = c13 / denom

            for c in columns:
                if c == "ratio":
                    if ratio_val is None:
                        continue
                    out[c].append(float(ratio_val))
                    continue

                v_raw = row.get(c, "")
                if v_raw is None or str(v_raw).strip() == "":
                    continue
                try:
                    out[c].append(float(v_raw))
                except Exception as e:
                    raise ValueError(f"Failed parsing float in {csv_path} column={c!r}: {v_raw!r}") from e

    # Ensure each requested column has some data so boxplot won't error.
    empties = [c for c, vs in out.items() if len(vs) == 0]
    if empties:
        raise ValueError(f"No numeric data found for columns {empties} in {csv_path}")

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Read one or more c12/c13 per-ROI solve output CSVs and plot boxplots per series. "
            "Each input CSV is treated as one series."
        )
    )
    p.add_argument(
        "csv",
        nargs="+",
        type=Path,
        help="One or more CSV files written by solve_c12_c13.py (per-ROI mode)",
    )
    p.add_argument(
        "--columns",
        type=_parse_columns,
        default=list(DEFAULT_COLUMNS),
        help=(
            "Comma-separated list of numeric columns to plot. Special name: 'ratio' = c13/(c12+c13). "
            f"(default: {','.join(DEFAULT_COLUMNS)})"
        ),
    )
    p.add_argument(
        "--names",
        type=_parse_names,
        default=None,
        help=(
            "Comma-separated series names (same count/order as input CSVs). "
            "Default: inferred from each filename stem."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./c12_c13_boxplots.png"),
        help=(
            "Save figure to this path (png/pdf/svg, etc). "
            "Default: ./c12_c13_boxplots.png"
        ),
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="If set, also show an interactive window after saving",
    )
    p.add_argument(
        "--figsize",
        type=str,
        default=None,
        help="Figure size in inches as W,H (e.g. 10,4). Default: chosen automatically.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    csv_paths: list[Path] = list(args.csv)
    columns: list[str] = list(args.columns)

    if args.names is None:
        names = [p.stem for p in csv_paths]
    else:
        names = list(args.names)
        if len(names) != len(csv_paths):
            raise ValueError(f"--names count ({len(names)}) must match number of CSVs ({len(csv_paths)})")

    # Read all series first.
    series: list[dict[str, list[float]]] = [
        _read_numeric_columns(p, columns=columns) for p in csv_paths
    ]

    ncols = len(columns)
    if args.figsize is None:
        figsize = (max(6.0, 1.8 * len(csv_paths)), 3.5 * ncols)
    else:
        w_str, h_str = (x.strip() for x in args.figsize.split(",", maxsplit=1))
        figsize = (float(w_str), float(h_str))

    fig, axes = plt.subplots(nrows=ncols, ncols=1, figsize=figsize, constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns, strict=True):
        data = [s[col] for s in series]
        # Hide default "fliers" (outlier markers) since we overlay all points ourselves.
        ax.boxplot(data, labels=names, showfliers=False)
        ax.set_ylabel(col)
        ax.grid(True, axis="y", alpha=0.25)

        # Overlay each ROI point (strip plot) so distribution is visible.
        for i, ys in enumerate(data, start=1):
            if len(ys) == 0:
                continue
            # Deterministic jitter based on index (no RNG).
            if len(ys) == 1:
                xs = np.array([float(i)], dtype=np.float64)
            else:
                xs = float(i) + np.linspace(-0.18, 0.18, num=len(ys), dtype=np.float64)
            ax.scatter(xs, ys, s=10, alpha=0.55, linewidths=0, zorder=3)

        if col == "ratio":
            ax.set_ylim(0.9, 1.1)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Wrote: {out_path}")
    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
