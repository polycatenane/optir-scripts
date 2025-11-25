#!/usr/bin/env python3
"""Create stacked histogram from two TIFF files."""
import argparse
import numpy as np
from tifffile import imread

import matplotlib.pyplot as plt

def read_values(path, threshold=1e-5):
    img = imread(path)
    arr = np.asarray(img, dtype=np.float32)
    flat = arr.ravel()
    vals = flat[flat > threshold]
    return vals

def main():
    parser = argparse.ArgumentParser(description="Stacked histogram of two TIFF files")
    parser.add_argument("tiff1", help="Path to first TIFF file")
    parser.add_argument("tiff2", help="Path to second TIFF file")
    parser.add_argument("-o","--out", default="stacked_histogram.png", help="Output figure path")
    parser.add_argument("--title", default="Normalized Treatment vs Control Histogram", help="Figure title")
    args = parser.parse_args()

    vals1 = read_values(args.tiff1)
    vals2 = read_values(args.tiff2)

    combined_data = np.concatenate([vals1, vals2])
    combined_bins = np.histogram_bin_edges(combined_data, bins='auto')  

    if vals1.size==0 and vals2.size==0:
        raise SystemExit("No values > threshold found in either file.")

    colors = ["steelblue","firebrick"]
    labels = ["Control", "AHA"]

    plt.figure(figsize=(8,6))
    plt.hist(vals1, bins=combined_bins, color=colors[0], label=labels[0], density=True, linewidth=0.5)
    plt.hist(vals2, bins=combined_bins, color=colors[1], label=labels[1], density=True, linewidth=0.5)

    plt.xlabel("Ratioed Intensity")
    plt.xlim(0, 0.7)
    plt.ylabel("Normalized Counts")
    plt.title(args.title)
    plt.legend(edgecolor="black")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved stacked histogram to {args.out}")

if __name__=='__main__':
    main()
