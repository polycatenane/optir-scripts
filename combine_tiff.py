#!/usr/bin/env python3
"""Combine two float32 TIFFs with elementwise multiply/divide and save float32."""
import argparse
import sys
from pathlib import Path
import numpy as np
from skimage import io

def parse_args():
    parser = argparse.ArgumentParser(
        description="Elementwise multiply or divide two float32 TIFFs and save result as float32 TIFF."
    )
    parser.add_argument("a", help="First input TIFF file (float32).")
    parser.add_argument("b", help="Second input TIFF file (float32).")
    parser.add_argument("output", nargs="?", help="Output TIFF path. If omitted, derived from inputs.", default=None)
    parser.add_argument("--op", choices=["mul", "div"], default="mul",
                        help="Operation: mul (multiply) or div (divide). Default: mul")
    parser.add_argument("--allow-broadcast", action="store_true",
                        help="Allow numpy broadcasting if shapes differ.")
    return parser.parse_args()

def read_tiff(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    img = io.imread(str(path))
    return img.astype(np.float32, copy=False)

def combine_arrays(a: np.ndarray, b: np.ndarray, op: str, allow_broadcast: bool) -> np.ndarray:
    if not allow_broadcast and a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}. Use --allow-broadcast to permit numpy broadcasting.")
    if op == "mul":
        res = a * b
    else:
        res = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return res.astype(np.float32, copy=False)

def main():
    args = parse_args()
    a_path = Path(args.a)
    b_path = Path(args.b)
    out_path = Path(args.output) if args.output else a_path.with_name(f"{a_path.stem}_{args.op}_{b_path.stem}.tif")
    try:
        a = read_tiff(a_path)
        b = read_tiff(b_path)
    except Exception as e:
        print(f"Failed reading inputs: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        res = combine_arrays(a, b, args.op, args.allow_broadcast)
    except Exception as e:
        print(f"Failed combining arrays: {e}", file=sys.stderr)
        sys.exit(1)
    io.imsave(str(out_path), res.astype(np.float32), check_contrast=False)
    print(f"Wrote {out_path} (shape={res.shape}, dtype=float32)")

if __name__ == "__main__":
    main()