#!/usr/bin/env python3
"""Convert CSV text-image files to 32-bit float TIFF files."""
import argparse
import sys
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from skimage import io

def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV image files to float32 TIFFs.")
    parser.add_argument("targets", nargs="+", help="Files and/or directories to scan (recursively).")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',').")
    parser.add_argument("--contains", default="", help="Only process files whose name contains this substring (case-sensitive).")
    parser.add_argument("--ext", default=".csv", help="File extension to look for (default: .csv)")
    parser.add_argument("--orig-dir", default="original_csvs", help="Directory name (under each file's parent) to move original CSVs into. Set to '' to disable moving.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing TIFFs instead of skipping.")
    return parser.parse_args()

def read_csv_image(path: Path, delimiter: str):
    try:
        df = pd.read_csv(path, header=None, delimiter=delimiter, dtype=float, engine="python")
    except Exception as e:
        raise RuntimeError(f"Failed reading {path}: {e}")
    arr = df.values.astype(np.float32, copy=False)
    return arr

def convert_file(path: Path, args):
    arr = read_csv_image(path, args.delimiter)
    out_path = path.with_suffix(".tif")
    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing {out_path}; use --overwrite to replace")
        return
    # save float32 tiff
    io.imsave(str(out_path), arr.astype(np.float32))
    print(f"Wrote {out_path} (shape={arr.shape}, dtype=float32)")
    # move original CSV into per-directory archive
    if args.orig_dir:
        dest_dir = path.parent / args.orig_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        # if destination exists, add timestamp to avoid overwrite
        if dest.exists():
            suffix = time.strftime(".%Y%m%d-%H%M%S")
            dest = dest.with_name(dest.stem + suffix + dest.suffix)
        shutil.move(str(path), str(dest))
        print(f"Moved original CSV to {dest}")

def main():
    args = parse_args()
    for t in args.targets:
        p = Path(t)
        if p.is_file():
            files = [p]
        else:
            files = list(p.rglob(f"*{args.ext}"))
        for f in files:
            if args.contains and args.contains not in f.name:
                continue
            try:
                convert_file(f, args)
            except Exception as e:
                print(f"Skipped {f}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()