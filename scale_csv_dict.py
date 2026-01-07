#!/usr/bin/env python3
import argparse
import json
import csv
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tifffile as tiff

try:
    import yaml  # optional
except ImportError:
    yaml = None

# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Scale numeric values in TIFF images using factors "
                "looked up from keys embedded in file names."
)
parser.add_argument("targets", nargs="+",
                    help="Directories whose top-level and immediate subdirectories will be scanned.")
parser.add_argument("--dict", required=True,
                    help="JSON or YAML file whose contents map "
                         "'key' (number before cm-1) -> scale factor.")
parser.add_argument("--op", choices=["mul", "div"], default="mul",
                    help="mul = multiply by factor, div = divide by it.")
parser.add_argument("--pattern",
                    default=r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*cm-1",
                    help="Regexp that captures the key from the file name.")
parser.add_argument("--orig-dir", default="unscaled_tiffs",
                    help="Directory name (inside each traversed directory) to move original TIFFs into. Set to '' to disable moving.")
parser.add_argument("--contains", default="AC",
                    help="Only process files whose *name* contains this "
                         "substring (case-sensitive). Default: 'AC'")
args = parser.parse_args()

# ---------- load mapping ----------
mapping_path = Path(args.dict)
if mapping_path.suffix.lower() in {".yaml", ".yml"}:
    if yaml is None:
        sys.exit("Install pyyaml or give a JSON file for --dict")
    with open(mapping_path, "rt") as f:
        raw_map = yaml.safe_load(f)
elif mapping_path.suffix.lower() == ".csv":
    raw_map = {}
    with open(mapping_path, "rt", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                continue
            k = row[0].strip()
            v = row[1].strip()
            raw_map[k] = v
else:  # assume JSON
    with open(mapping_path, "rt") as f:
        raw_map = json.load(f)

# Normalize keys to rounded integer strings and ensure values are floats
SCALE_MAP = {}
for k, v in raw_map.items():
    try:
        key_num = float(k)
        key_norm = str(int(round(key_num)))
    except Exception:
        key_norm = str(k)
    try:
        SCALE_MAP[key_norm] = float(v)
    except Exception:
        SCALE_MAP[key_norm] = v

regex = re.compile(args.pattern)

def get_factor(fname: str) -> float:
    """Find key in filename and return its factor."""
    m = regex.search(fname)
    if not m:
        raise KeyError(f"No key found in '{fname}'")
    key = m.group(1)
    if key not in SCALE_MAP:
        raise KeyError(f"Key '{key}' not found in mapping file")
    return float(SCALE_MAP[key])

def scale_image(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale numeric image ndarray and return float32 result."""
    arr = img.astype(np.float32, copy=True)
    if args.op == "mul":
        arr = arr * factor
    else:
        arr = arr / factor
    return arr.astype(np.float32)

def process(path: Path):
    factor = get_factor(path.name)
    print(f"{path}: factor={factor} ({'×' if args.op=='mul' else '÷'})")
    img = tiff.imread(str(path))
    scaled = scale_image(img, factor)
    # Move original TIFF into per-directory archive if requested
    if args.orig_dir:
        dest_dir = path.parent / args.orig_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        # avoid overwrite at destination by adding timestamp if needed
        if dest.exists():
            suffix = time.strftime(".%Y%m%d-%H%M%S")
            dest = dest.with_name(dest.stem + suffix + dest.suffix)
        shutil.move(str(path), str(dest))
        print(f"Moved original TIFF to {dest}")
    # write scaled TIFF (float32)
    tiff.imwrite(str(path), scaled.astype(np.float32))

# ---------- main ----------
for t in args.targets:
    p = Path(t)
    files = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        # collect TIFFs in the directory itself (top-level)
        files.extend(list(p.glob("*.tif")) + list(p.glob("*.tiff")))
        # collect TIFFs in immediate subdirectories (one level deep)
        for sd in (d for d in p.iterdir() if d.is_dir()):
            files.extend(list(sd.glob("*.tif")) + list(sd.glob("*.tiff")))
    else:
        continue
    for f in files:
        if args.contains and args.contains not in f.name:
            continue
        try:
            process(f)
        except Exception as e:
            print(f"Skipped {f}: {e}", file=sys.stderr)