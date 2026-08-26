#!/usr/bin/env python3
"""Rectify noise by subtracting mean of pixelwise product of two float32 TIFFs.

Given two input TIFF images with float32 values and identical shape, this script:

1. Multiplies them element-wise.
2. Computes the mean value of the resulting product image.
3. Subtracts this mean from every pixel of the first input image.
4. Writes the result as a new float32 TIFF.

Usage
-----

    rectify_noise.py input.tif reference.tif [output.tif]

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
            "Rectify noise by subtracting the mean of the pixelwise product "
            "of two float32 TIFF images from the first image."
        )
    )
    parser.add_argument("input", help="First input TIFF file (float32).")
    parser.add_argument(
        "reference",
        help="Second input TIFF file (float32) to multiply with the first.",
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

    if img.shape != ref.shape:
        print(
            f"Shape mismatch: input shape={img.shape}, reference shape={ref.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Element-wise product and mean
    product = img * ref
    mean_val = float(np.mean(product))

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
        f"mean_product={mean_val})"
    )


if __name__ == "__main__":
    main()

