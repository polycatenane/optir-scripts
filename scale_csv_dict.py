#!/usr/bin/env python3
import argparse, json, re, shutil, sys, time
from pathlib import Path

import pandas as pd

try:
    import yaml  # optional
except ImportError:
    yaml = None

# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Scale numeric values in CSVs using factors "
                "looked up from keys embedded in file names."
)
parser.add_argument("targets", nargs="+",
                    help="Files and/or directories to scan (recursively).")
parser.add_argument("--dict", required=True,
                    help="JSON or YAML file whose contents map "
                         "'key' (number before invcm) -> scale factor.")
parser.add_argument("--op", choices=["mul", "div"], default="mul",
                    help="mul = multiply by factor, div = divide by it.")
parser.add_argument("--pattern",
                    default=r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*IR",
                    help="Regexp that captures the key from the file name.")
parser.add_argument("--backup-suffix",
                    default=time.strftime(".bak_%Y%m%d-%H%M%S"),
                    help="Suffix appended to original file before overwriting.")
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
        SCALE_MAP = yaml.safe_load(f)
else:  # assume JSON
    with open(mapping_path, "rt") as f:
        SCALE_MAP = json.load(f)

# Normalize keys to string for flexible matching
SCALE_MAP = {str(k): v for k, v in SCALE_MAP.items()}

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

def scale_frame(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns
    if args.op == "mul":
        df[num_cols] = df[num_cols] * factor
    else:
        df[num_cols] = df[num_cols] / factor
    return df

def process(path: Path):
    factor = get_factor(path.name)
    print(f"{path}: factor={factor} ({'×' if args.op=='mul' else '÷'})")
    df = pd.read_csv(path)
    df = scale_frame(df, factor)
    backup = path.with_suffix(path.suffix + args.backup_suffix)
    shutil.copy2(path, backup)
    df.to_csv(path, index=False)

# ---------- main ----------
for t in args.targets:
    p = Path(t)
    files = [p] if p.is_file() else p.rglob("*.csv")
    for f in files:
        if args.contains and args.contains not in f.name:
            continue                    # <── filter by token
        try:
            process(f)
        except Exception as e:
            print(f"Skipped {f}: {e}", file=sys.stderr)
