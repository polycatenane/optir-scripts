#!/usr/bin/env python3
"""Two-sample t-test on ROI measurements exported by segment_from_mask.py.

This script expects CSV inputs matching the output of segment_from_mask.py,
which contains one row per ROI and columns like: area, mean_intensity, etc.

Example
  ./two_sample_ttest.py a_rois.csv b_rois.csv --column mean_intensity
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GroupSummary:
    n: int
    mean: float
    std: float  # sample std (ddof=1); NaN if n < 2


def _positive_float(value: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number > 0")
    return v


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a two-sample t-test on a numeric column from two ROI CSVs "
            "(as produced by segment_from_mask.py)."
        )
    )
    p.add_argument("sample_a_csv", type=Path, help="CSV for sample/group A")
    p.add_argument("sample_b_csv", type=Path, help="CSV for sample/group B")
    p.add_argument(
        "--column",
        default="mean_intensity",
        help="Numeric column to compare. Default: mean_intensity",
    )
    p.add_argument(
        "--equal-var",
        action="store_true",
        help="Assume equal variance (Student t-test). Default is Welch (unequal variance).",
    )
    p.add_argument(
        "--alternative",
        choices=("two-sided", "less", "greater"),
        default="two-sided",
        help="Alternative hypothesis. Default: two-sided",
    )
    p.add_argument(
        "--min-area",
        type=_positive_float,
        default=None,
        help="Optional: keep only ROIs with area >= this value (uses 'area' column).",
    )
    p.add_argument(
        "--max-area",
        type=_positive_float,
        default=None,
        help="Optional: keep only ROIs with area <= this value (uses 'area' column).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional: write a one-row CSV summary to this path.",
    )
    return p.parse_args(argv)


def _safe_float(s: str) -> float | None:
    s2 = s.strip()
    if s2 == "":
        return None
    try:
        v = float(s2)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def read_numeric_column(
    csv_path: Path,
    *,
    column: str,
    min_area: float | None,
    max_area: float | None,
) -> list[float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        if column not in r.fieldnames:
            fields = ", ".join(r.fieldnames)
            raise ValueError(f"Column '{column}' not found in {csv_path}. Columns: {fields}")

        needs_area = (min_area is not None) or (max_area is not None)
        if needs_area and "area" not in r.fieldnames:
            raise ValueError(
                f"Area filtering requested but column 'area' not found in {csv_path}."
            )

        out: list[float] = []
        for row in r:
            if needs_area:
                a = _safe_float(row.get("area", ""))
                if a is None:
                    continue
                if min_area is not None and a < min_area:
                    continue
                if max_area is not None and a > max_area:
                    continue

            v = _safe_float(row.get(column, ""))
            if v is None:
                continue
            out.append(v)

    return out


def summarize(x: Sequence[float]) -> GroupSummary:
    n = len(x)
    if n == 0:
        return GroupSummary(n=0, mean=float("nan"), std=float("nan"))
    mean = float(sum(x) / n)
    if n < 2:
        return GroupSummary(n=n, mean=mean, std=float("nan"))
    ss = sum((xi - mean) ** 2 for xi in x)
    std = float(math.sqrt(ss / (n - 1)))
    return GroupSummary(n=n, mean=mean, std=std)


def _welch_df(*, s1: float, n1: int, s2: float, n2: int) -> float:
    # Welch–Satterthwaite equation; expects sample stds.
    v1 = (s1 * s1) / n1
    v2 = (s2 * s2) / n2
    num = (v1 + v2) ** 2
    den = (v1 * v1) / (n1 - 1) + (v2 * v2) / (n2 - 1)
    return float(num / den)


def _student_df(*, n1: int, n2: int) -> float:
    return float(n1 + n2 - 2)


def run_ttest(
    a: Sequence[float],
    b: Sequence[float],
    *,
    equal_var: bool,
    alternative: str,
) -> tuple[float, float, float]:
    try:
        from scipy import stats  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "scipy is required for p-values (ttest). Install it with: pip install scipy"
        ) from e

    res = stats.ttest_ind(
        a,
        b,
        equal_var=equal_var,
        alternative=alternative,
        nan_policy="omit",
    )
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)

    sa = summarize(a)
    sb = summarize(b)
    if equal_var:
        df = _student_df(n1=sa.n, n2=sb.n) if (sa.n >= 2 and sb.n >= 2) else float("nan")
    else:
        df = (
            _welch_df(s1=sa.std, n1=sa.n, s2=sb.std, n2=sb.n)
            if (sa.n >= 2 and sb.n >= 2)
            else float("nan")
        )
    return t_stat, p_val, df


def write_summary_csv(
    out_path: Path,
    *,
    column: str,
    equal_var: bool,
    alternative: str,
    a_path: Path,
    b_path: Path,
    a_sum: GroupSummary,
    b_sum: GroupSummary,
    t_stat: float,
    p_val: float,
    df: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "column",
        "equal_var",
        "alternative",
        "sample_a_csv",
        "sample_b_csv",
        "n_a",
        "mean_a",
        "std_a",
        "n_b",
        "mean_b",
        "std_b",
        "t_stat",
        "p_value",
        "df",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(
            {
                "column": column,
                "equal_var": str(bool(equal_var)),
                "alternative": alternative,
                "sample_a_csv": str(a_path),
                "sample_b_csv": str(b_path),
                "n_a": a_sum.n,
                "mean_a": a_sum.mean,
                "std_a": a_sum.std,
                "n_b": b_sum.n,
                "mean_b": b_sum.mean,
                "std_b": b_sum.std,
                "t_stat": t_stat,
                "p_value": p_val,
                "df": df,
            }
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    a = read_numeric_column(
        args.sample_a_csv,
        column=args.column,
        min_area=args.min_area,
        max_area=args.max_area,
    )
    b = read_numeric_column(
        args.sample_b_csv,
        column=args.column,
        min_area=args.min_area,
        max_area=args.max_area,
    )

    a_sum = summarize(a)
    b_sum = summarize(b)
    if a_sum.n < 2 or b_sum.n < 2:
        print(
            f"Need at least 2 ROIs per group after filtering; got n_a={a_sum.n}, n_b={b_sum.n}",
            file=sys.stderr,
        )
        return 2

    t_stat, p_val, df = run_ttest(
        a,
        b,
        equal_var=bool(args.equal_var),
        alternative=str(args.alternative),
    )

    test_name = "Student" if args.equal_var else "Welch"
    print(f"Column: {args.column}")
    print(f"Test: {test_name} two-sample t-test (alternative={args.alternative})")
    print(f"A: n={a_sum.n} mean={a_sum.mean:.6g} std={a_sum.std:.6g} file={args.sample_a_csv}")
    print(f"B: n={b_sum.n} mean={b_sum.mean:.6g} std={b_sum.std:.6g} file={args.sample_b_csv}")
    print(f"t={t_stat:.6g} df={df:.6g} p={p_val:.6g}")

    if args.out_csv is not None:
        write_summary_csv(
            args.out_csv,
            column=args.column,
            equal_var=bool(args.equal_var),
            alternative=str(args.alternative),
            a_path=args.sample_a_csv,
            b_path=args.sample_b_csv,
            a_sum=a_sum,
            b_sum=b_sum,
            t_stat=t_stat,
            p_val=p_val,
            df=df,
        )
        print(f"Wrote summary CSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

