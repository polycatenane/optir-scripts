#!/usr/bin/env python3
"""Compute Cohen's d and t-test for two TIFF files (values > threshold)."""

import argparse
import numpy as np
from tifffile import imread

from scipy import stats

def read_values(path, threshold=1e-5):
    img = imread(path)
    arr = np.asarray(img, dtype=np.float64)
    flat = arr.ravel()
    vals = flat[flat > threshold]
    return vals

def cohens_d(x, y):
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return np.nan
    mx = x.mean()
    my = y.mean()
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    pooled_var = ((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2)
    pooled_sd = np.sqrt(pooled_var)
    if pooled_sd == 0:
        return np.nan
    return (mx - my) / pooled_sd

def main():
    parser = argparse.ArgumentParser(description="Cohen's d and t-test for two TIFF files")
    parser.add_argument("tiff1", help="Path to first TIFF file")
    parser.add_argument("tiff2", help="Path to second TIFF file")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Intensity threshold")
    parser.add_argument("--equal-var", action="store_true", help="Assume equal variances for t-test")
    args = parser.parse_args()

    vals1 = read_values(args.tiff1, threshold=args.threshold)
    vals2 = read_values(args.tiff2, threshold=args.threshold)

    print(f"n1={vals1.size}, n2={vals2.size}")
    if vals1.size == 0 or vals2.size == 0:
        print("One or both inputs have no values above threshold; aborting.")
        return

    print(f"mean1={vals1.mean():.6g}, std1={vals1.std(ddof=1):.6g}")
    print(f"mean2={vals2.mean():.6g}, std2={vals2.std(ddof=1):.6g}")

    d = cohens_d(vals1, vals2)
    tstat, pval = stats.ttest_ind(vals1, vals2, equal_var=args.equal_var, nan_policy='omit')
 
    print(f"Cohen's d (pooled sd) = {d:.6g}")
    print(f"t-statistic = {tstat:.6g}, p-value = {pval:.6g}")
 
if __name__ == "__main__":
    main()