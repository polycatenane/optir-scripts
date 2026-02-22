#!/usr/bin/env python3

"""
Assemble a hyperstack image from all TIFF images in a directory
whose filename contains the substring "AC". The stack is ordered by
the numeric value immediately preceding "cm-1" in each filename,
sorted in ascending order.

Example filenames:
  sample_850cm-1_AC.tif
  AC_1234cm-1_region1.tiff

Usage:
  python hyperstack_from_ac_tiffs.py /path/to/dir -o output.tif
"""

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile


NUMBER_CM1_PATTERN = re.compile(r"(\d+)\s*cm-1", re.IGNORECASE)


def extract_wavenumber(path: Path) -> int:
    """Extract the integer immediately preceding 'cm-1' in the filename.

    Returns:
        int: The extracted wavenumber.

    Raises:
        ValueError: If no such number can be found.
    """
    match = NUMBER_CM1_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"No '<number>cm-1' pattern found in filename: {path.name}")
    return int(match.group(1))


def collect_ac_tiffs(directory: Path):
    """Find all TIFF files in the directory whose name contains 'AC'
    (case-insensitive) and that contain a '<number>cm-1' pattern.

    Returns:
        list[tuple[int, Path]]: List of (wavenumber, path) pairs.
    """
    candidates = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if "ac" not in lower:
            continue
        if not (lower.endswith(".tif") or lower.endswith(".tiff")):
            continue
        try:
            wn = extract_wavenumber(path)
        except ValueError:
            # Skip files without a valid wavenumber
            continue
        candidates.append((wn, path))

    if not candidates:
        raise RuntimeError("No matching AC TIFF files with '<number>cm-1' in the name found.")

    # Sort by wavenumber ascending
    candidates.sort(key=lambda x: x[0])
    return candidates


def build_hyperstack(ordered_paths: list[Path]) -> np.ndarray:
    """Load images and stack them along a new first axis.

    All images must have the same shape and dtype.
    """
    images = []
    first_shape = None
    first_dtype = None

    for path in ordered_paths:
        img = tifffile.imread(path)
        if first_shape is None:
            first_shape = img.shape
            first_dtype = img.dtype
        else:
            if img.shape != first_shape:
                raise RuntimeError(
                    f"Image {path.name} has shape {img.shape}, "
                    f"but expected {first_shape}"
                )
            if img.dtype != first_dtype:
                raise RuntimeError(
                    f"Image {path.name} has dtype {img.dtype}, "
                    f"but expected {first_dtype}"
                )
        images.append(img)

    stack = np.stack(images, axis=0)
    return stack


def save_hyperstack(stack: np.ndarray, output_path: Path):
    """Save the stack as a multi-page TIFF that ImageJ/Fiji will open as a stack."""
    # tifffile will write a multi-page TIFF for a 3D array when imagej=True
    tifffile.imwrite(
        output_path,
        stack,
        imagej=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a hyperstack from AC TIFF images in a directory, "
            "ordered by the <number> before 'cm-1' in each filename."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing TIFF images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output TIFF path (default: 'hyperstack_AC.tif' in the directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory: Path = args.directory

    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    try:
        wn_and_paths = collect_ac_tiffs(directory)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    ordered_paths = [p for _, p in wn_and_paths]
    stack = build_hyperstack(ordered_paths)

    if args.output is not None:
        output = args.output
    else:
        output = directory / "hyperstack_AC.tif"

    save_hyperstack(stack, output)
    print(f"Wrote hyperstack with {stack.shape[0]} planes to: {output}")


if __name__ == "__main__":
    main()

