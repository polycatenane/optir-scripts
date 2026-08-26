#!/usr/bin/env python3
"""Batch-combine float32 TIFFs with a per-directory mask TIFF.

For each directory under the given targets, this script looks for exactly one
TIFF whose filename contains the substring "mask" (case-insensitive) and
treats that as the mask image for that directory. All other TIFFs in that
directory are combined elementwise with that mask, using the same arithmetic
and broadcasting rules as [`combine_tiff.py`](combine_tiff.py:1).

Outputs are written next to each input image as
<image_stem>_<op>_<mask_stem>.tif. After successful combination, the original
image TIFFs and their mask TIFF are moved into an archive subdirectory so
that only the newly created files remain.
"""

import argparse
import sys
import time
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import tifffile as tiff


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch elementwise multiply/divide float32 TIFFs by an automatically "
            "detected per-directory mask TIFF (filename contains 'mask'), "
            "using combine_tiff.py logic."
        )
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="Files and/or directories to scan recursively for TIFFs.",
    )
    parser.add_argument(
        "--op",
        choices=["mul", "div"],
        default="mul",
        help="Operation: mul (multiply) or div (divide). Default: mul",
    )
    parser.add_argument(
        "--allow-broadcast",
        action="store_true",
        help="Allow numpy broadcasting if shapes differ (same as combine_tiff.py).",
    )
    parser.add_argument(
        "--contains",
        default="AC",
        help=(
            "Only process TIFFs whose *name* contains this substring "
            "(case-sensitive), excluding the mask itself."
        ),
    )
    parser.add_argument(
        "--ext",
        default=".tif",
        help="File extension to look for (default: .tif). Can be .tif or .tiff",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output TIFFs instead of skipping.",
    )
    parser.add_argument(
        "--orig-dir",
        default="original_tiffs",
        help=(
            "Directory name (inside each image's directory) to move original "
            "TIFFs into after combination. Set to '' to disable moving."
        ),
    )
    return parser.parse_args()


def read_tiff(path: Path) -> np.ndarray:
    """Read TIFF as float32 (same helper logic as in combine_tiff.py)."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    img = tiff.imread(str(path))
    return img.astype(np.float32, copy=False)


def combine_arrays(a: np.ndarray, b: np.ndarray, op: str,
allow_broadcast: bool) -> np.ndarray:
    """Elementwise combine two arrays, mirroring combine_tiff.py behavior."""
    if not allow_broadcast and a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. "
            "Use --allow-broadcast to permit numpy broadcasting."
        )
    if op == "mul":
        res = a * b
    else:
        # avoid division by zero
        res = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return res.astype(np.float32, copy=False)


def gather_tiffs(targets, ext: str, orig_dir_name: str):
    """Collect all TIFF files under given targets (recursive),
    explicitly skipping any files that live under the orig-dir archive
    directory (e.g. 'original_tiffs').
    """
    ext = ext.lower()
    files = []
    for t in targets:
        p = Path(t)
        if p.is_file():
            # Skip anything already inside the archive directory
            if orig_dir_name and orig_dir_name in p.parts:
                continue
            if p.suffix.lower() == ext or (ext == ".tif" and 
            p.suffix.lower() == ".tiff"):
                files.append(p)
        else:
            # Walk recursively but ignore any subtrees whose path contains orig_dir_name
            for f in p.rglob("*"):
                if not f.is_file():
                    continue
                if orig_dir_name and orig_dir_name in f.parts:
                    continue
                suf = f.suffix.lower()
                if suf == ext or (ext == ".tif" and suf == ".tiff"):
                    files.append(f)
    return files


def main():
    args = parse_args()
    # Gather all TIFFs and group them by parent directory
    all_files = gather_tiffs(args.targets, args.ext, args.orig_dir)
    dir_masks = {}
    dir_images = defaultdict(list)

    for f in all_files:
        parent = f.parent
        name_lower = f.name.lower()
        if "mask" in name_lower:
            if parent in dir_masks and dir_masks[parent] != f:
                print(f"Directory {parent} has multiple maskcandidates; skipping {f}", file=sys.stderr)
                continue
            dir_masks[parent] = f
        else:
            dir_images[parent].append(f)

    # For each directory with a mask, combine that mask with all non-mask images
    for d, mask_path in dir_masks.items():
        images = dir_images.get(d, [])
        if not images:
            continue

        try:
            mask = read_tiff(mask_path)
        except Exception as e:
            print(f"Failed reading mask TIFF {mask_path}: {e}", file=sys.stderr)
            continue

        processed_any = False

        for img_path in images:
            if args.contains and args.contains not in img_path.name:
                continue
            try:
                img = read_tiff(img_path)
            except Exception as e:
                print(f"Skipped {img_path}: failed reading image: {e}", file=sys.stderr)
                continue

            out_path = img_path.with_name(
                f"{img_path.stem}_{args.op}_{mask_path.stem}.tif"
            )
            if out_path.exists() and not args.overwrite:
                print(f"Skipping existing {out_path}; use --overwrite to replace")
                continue

            try:
                res = combine_arrays(img, mask, args.op, args.allow_broadcast)
            except Exception as e:
                print(f"Skipped {img_path}: failed combining with mask: {e}", file=sys.stderr)
                continue

            try:
                tiff.imwrite(str(out_path), res.astype(np.float32))
            except Exception as e:
                print(f"Failed writing {out_path}: {e}", file=sys.stderr)
                continue

            print(f"Wrote {out_path} (shape={res.shape}, dtype=float32)")

            # Move only this original image into per-directory archive if requested
            if args.orig_dir:
                dest_dir = img_path.parent / args.orig_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / img_path.name
                if dest.exists():
                    suffix = time.strftime(".%Y%m%d-%H%M%S")
                    dest = dest.with_name(dest.stem + suffix + dest.suffix)
                try:
                    shutil.move(str(img_path), str(dest))
                    print(f"Moved original TIFF {img_path} to {dest}")
                except Exception as e:
                    print(f"Failed moving {img_path} to {dest}: {e}", file=sys.stderr)

            processed_any = True

        # After processing all images in this directory, move the mask itself
        # but only if at least one image was successfully combined
        if args.orig_dir and processed_any:
            dest_dir = mask_path.parent / args.orig_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / mask_path.name
            if dest.exists():
                suffix = time.strftime(".%Y%m%d-%H%M%S")
                dest = dest.with_name(dest.stem + suffix + dest.suffix)
            try:
                shutil.move(str(mask_path), str(dest))
                print(f"Moved mask TIFF {mask_path} to {dest}")
            except Exception as e:
                print(f"Failed moving mask {mask_path} to {dest}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()