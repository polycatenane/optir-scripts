#!/usr/bin/env python3
"""Baseline-correct by subtracting the average intensity of a reference TIFF.

Given two input TIFF images with float32 values, this script:

1. Computes the mean intensity of the reference image.
2. Subtracts this mean from every pixel of the first input image.
3. Writes the result as a new float32 TIFF.

Usage
-----

    baseline_correct.py input.tif reference.tif [output.tif]

If output is omitted, ``_rectified.tif`` is appended to the stem of ``input.tif``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline-correct an input TIFF by subtracting the mean intensity "
            "of a reference TIFF."
        )
    )
    parser.add_argument("input", help="First input TIFF file (float32).")
    parser.add_argument(
        "reference",
        help="Reference TIFF file (float32); its mean intensity is subtracted from the input.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "Output TIFF path. If omitted, appends _rectified.tif to the "
            "first input's stem."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inp_path = Path(args.input)
    ref_path = Path(args.reference)

    if not inp_path.exists():
        print(f"Input not found: {inp_path}", file=sys.stderr)
        sys.exit(1)
    if not ref_path.exists():
        print(f"Reference not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    # Load images as float32
    img = tiff.imread(str(inp_path)).astype(np.float32)
    ref = tiff.imread(str(ref_path)).astype(np.float32)

    # Use the reference's average intensity as the baseline value.
    mean_val = float(np.mean(ref))

    # Subtract mean from the first input image
    rectified = img - mean_val

    # Determine output path
    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = inp_path.with_name(inp_path.stem + "_rectified.tif")

    tiff.imwrite(str(out_path), rectified.astype(np.float32))

    print(
        f"Wrote {out_path} (shape={rectified.shape}, dtype=float32, "
        f"mean_reference={mean_val})"
    )


if __name__ == "__main__":
    main()
