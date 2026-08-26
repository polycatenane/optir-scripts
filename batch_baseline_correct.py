#!/usr/bin/env python3
"""Batch baseline-correction for a directory of TIFF images.

For each TIFF file in an input directory, this script:

1. Computes the mean intensity of a reference TIFF (float32).
2. Subtracts this mean from every pixel of the input image.
3. Writes the corrected image as a new float32 TIFF with ``_rectified`` suffix.
4. Moves the original input TIFF into a separate directory.

Usage
-----

    batch_baseline_correct.py INPUT_DIR reference.tif [--moved-dir ORIGINALS]

By default, originals are moved into ``INPUT_DIR/baseline_noise``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-rectify noise for all TIFF images in a directory using a "
            "single reference TIFF."
        )
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing input TIFF files (float32).",
    )
    parser.add_argument(
        "reference",
        help="Reference TIFF file (float32); its mean intensity is subtracted from each input image.",
    )
    parser.add_argument(
        "--moved-dir",
        "-m",
        default="baseline_noise",
        help=(
            "Directory to move original TIFFs into. If relative, it is created "
            "inside INPUT_DIR. Default: 'baseline_noise'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    ref_path = Path(args.reference)

    if not input_dir.is_dir():
        print(f"Input directory not found or not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not ref_path.exists():
        print(f"Reference not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve moved/originals directory
    moved_dir = Path(args.moved_dir)
    if not moved_dir.is_absolute():
        moved_dir = input_dir / moved_dir
    moved_dir.mkdir(parents=True, exist_ok=True)

    # Load reference image once and compute baseline value
    ref_img = tiff.imread(str(ref_path)).astype(np.float32)
    mean_val = float(np.mean(ref_img))

    # Collect input TIFF files (non-recursive)
    tiff_paths = []
    for pattern in ("*.tif", "*.tiff"):
        tiff_paths.extend(input_dir.glob(pattern))

    # Remove duplicates and sort
    tiff_paths = sorted(set(tiff_paths))

    # Only process files containing 'AC' in the filename
    tiff_paths = [p for p in tiff_paths if "AC" in p.name]

    # Optionally avoid processing the reference if it's inside input_dir
    ref_resolved = ref_path.resolve()
    tiff_paths = [p for p in tiff_paths if p.resolve() != ref_resolved]

    if not tiff_paths:
        print(f"No TIFF files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    for inp_path in tiff_paths:
        # Load input as float32
        img = tiff.imread(str(inp_path)).astype(np.float32)

        # Subtract mean from the input image
        rectified = img - mean_val

        # Output path in input_dir with _rectified suffix
        out_path = inp_path.with_name(inp_path.stem + "_rectified.tif")

        tiff.imwrite(str(out_path), rectified.astype(np.float32))

        # Move original file to moved_dir
        dest_path = moved_dir / inp_path.name
        inp_path.rename(dest_path)

        print(
            f"Processed {inp_path.name} -> {out_path.name}; "
            f"moved original to {dest_path} (mean_reference={mean_val})"
        )


if __name__ == "__main__":
    main()
