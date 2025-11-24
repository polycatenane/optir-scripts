#!/usr/bin/env python3
"""Create binary mask TIFF from float32 TIFF using Li threshold and cleanup."""
import argparse
import sys
from pathlib import Path
import numpy as np
from skimage import io, filters, morphology

def parse_args():
    parser = argparse.ArgumentParser(description="Create masked TIFF (float32 0/1) using Li threshold.")
    parser.add_argument("input", help="Input TIFF file (float32).")
    parser.add_argument("output", nargs="?", help="Output TIFF path. If omitted, adds _mask.tif", default=None)
    parser.add_argument("--area-threshold", type=int, default=64, help="Fill holes smaller than this (pixels). Default: 64")
    parser.add_argument("--min-size", type=int, default=6, help="Remove objects smaller than this (pixels). Default: 6")
    parser.add_argument("--invert", action="store_true", help="Invert mask (mask where value <= threshold).")
    return parser.parse_args()

def make_mask(img: np.ndarray, area_threshold:int, min_size: int, invert: bool) -> np.ndarray:
    # compute Li threshold (works on float images)
    thresh = filters.threshold_li(img)
    if invert:
        mask = img <= thresh
    else:
        mask = img > thresh
    # ensure boolean
    mask = mask.astype(bool)
    # remove small holes and objects
    mask = morphology.remove_small_holes(mask, area_threshold=area_threshold)
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    # convert to float32 0.0/1.0
    return mask.astype(np.float32)

def main():
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_mask.tif")
    img = io.imread(str(inp)).astype(np.float32)
    mask = make_mask(img, args.area_threshold, args.min_size, args.invert)
    io.imsave(str(out), mask.astype(np.float32), check_contrast=False)
    print(f"Wrote {out} (shape={mask.shape}, dtype=float32)")

if __name__ == "__main__":
    main()