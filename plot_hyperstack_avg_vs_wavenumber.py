#!/usr/bin/env python3

"""Plot average intensity per frame of a hyperstack versus wavenumber.

This is designed to work with hyperstacks created by
"hyperstack_from_ac_tiffs.py", which stack TIFFs containing "AC" in the
filename and sort them by the number before "cm-1".

Assumptions:
  * The hyperstack TIFF is located in the same directory as the original
    AC TIFFs that were used to build it.
  * The order of planes in the hyperstack matches the order used when
    stacking (sorted by wavenumber).

Usage:
  python plot_hyperstack_avg_vs_wavenumber.py /path/to/dir/hyperstack_AC.tif \
      -o avg_vs_wavenumber.png \
      --extra-data extra_series.txt \
      --legend-main "Average intensity" \
      --legend-extra "Experimental series"

When multiple extra series are provided, pass one legend per series, e.g.:
  --extra-data a.txt --extra-data b.txt --legend-extra "A" "B"

`--extra-data` can be either:
  * A text-like file (one numeric value per line / column), or
  * A TIFF hyperstack (".tif"/".tiff"), in which case the script will
    compute a second set of frame-wise means from that hyperstack and
    plot it against its *own* wavenumber axis. The extra hyperstack may
    therefore have a different number of frames than the main one.
"""

import argparse
from pathlib import Path

import io

import matplotlib.pyplot as plt
import numpy as np
import tifffile

try:
    # Optional dependency for baseline correction
    from pybaselines import Baseline
except ImportError:  # pragma: no cover - optional dependency
    Baseline = None

from hyperstack_from_ac_tiffs import collect_ac_tiffs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot average intensity per frame of a hyperstack TIFF "
            "versus wavenumber. Assumes the hyperstack is stored in the "
            "same directory as the original AC TIFFs used to build it."
        )
    )
    parser.add_argument(
        "hyperstack",
        type=Path,
        help="Path to hyperstack TIFF (e.g., /path/to/dir/hyperstack_AC.tif).",
    )
    parser.add_argument(
        "--extra-data",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional additional data series to plot. May be provided multiple times. "
            "Each value must be a path to either: "
            "(1) a text-like file (one numeric value per line / single column) with the same "
            "length as the number of frames after masking, or "
            "(2) a .tif/.tiff hyperstack, in which case frame-wise means are computed and plotted "
            "against that hyperstack's own wavenumber axis."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the plot image (PNG). "
            "Default: 'avg_intensity_vs_wavenumber.png' next to the hyperstack."
        ),
    )
    parser.add_argument(
        "--legend-main",
        type=str,
        default=None,
        help=(
            "Legend label for the main series. Default: the hyperstack filename (stem)."
        ),
    )
    parser.add_argument(
        "--legend-extra",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Legend label(s) for the extra series when --extra-data is provided. "
            "Provide one legend per extra series, e.g. --legend-extra L1 L2 L3. "
            "Default: each extra series uses its filename (stem)."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window (useful for batch scripts).",
    )
    parser.add_argument(
        "--airpls-baseline",
        action="store_true",
        help=(
            "If set, apply an airPLS baseline correction (via pybaselines) "
            "to the per-frame mean intensities before plotting. Requires "
            "the 'pybaselines' package to be installed."
        ),
    )
    parser.add_argument(
        "--asls-baseline",
        action="store_true",
        help=(
            "If set, apply an ASLS (asymmetric least squares) baseline "
            "correction (via pybaselines) to the per-frame mean intensities "
            "before plotting. Requires the 'pybaselines' package to be installed."
        ),
    )
    return parser.parse_args()


def load_hyperstack(path: Path) -> np.ndarray:
    """Load a hyperstack TIFF as a NumPy array.

    Returns a NumPy array with at least one axis; the first axis is
    interpreted as the frame index.
    """
    stack = tifffile.imread(path)
    if stack.ndim < 3:
        # Allow (frames, rows, cols) or more-dimensional data, but
        # require at least 3D for image data.
        raise RuntimeError(
            f"Expected at least 3D hyperstack data, got shape {stack.shape}"
        )
    return stack


def compute_frame_means(stack: np.ndarray) -> np.ndarray:
    """Compute average intensity per frame for a hyperstack.

    The first axis is treated as the frame index; all remaining axes
    are averaged over.
    """
    # Reshape each frame to a flat vector and take mean along pixels
    n_frames = stack.shape[0]
    flat = stack.reshape(n_frames, -1)
    means = flat.mean(axis=1)
    return means


def maybe_baseline_correct(y: np.ndarray, *, airpls: bool, asls: bool) -> np.ndarray:
    """Optionally baseline-correct a 1D series using pybaselines.

    Returns y unchanged if no baseline option was selected.
    """
    if not (airpls or asls):
        return y

    if airpls and asls:
        raise SystemExit("Choose only one of --airpls-baseline or --asls-baseline")

    if Baseline is None:
        raise SystemExit(
            "A baseline correction was requested but 'pybaselines' is not installed. "
            "Install it with `pip install pybaselines`."
        )

    baseline_fitter = Baseline()
    if airpls:
        baseline, _ = baseline_fitter.airpls(y, lam=1e1)
    else:
        baseline, _ = baseline_fitter.asls(y, lam=1e1, p=0.001)

    return y - baseline


def load_extra_series(path: Path, expected_len: int) -> np.ndarray:
    """Robustly load an extra y-series from a text-like file.

    Strategy:
      1. Read raw bytes.
      2. Try UTF-8 decode; on failure, fall back to latin-1.
      3. Parse numeric values with ``numpy.loadtxt`` from an in-memory
         text buffer.

    A short diagnostic of the first bytes is printed to help debug
    encoding/content issues.
    """
    raw = path.read_bytes()

    # Print a small diagnostic snippet for debugging
    preview = raw[:128]
    print(
        f"[diagnostic] extra-data file '{path}' size={len(raw)} bytes, "
        f"first bytes={preview!r}"
    )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        print(
            f"[diagnostic] UTF-8 decode failed for '{path}', "
            "falling back to latin-1."
        )
        text = raw.decode("latin-1")

    buf = io.StringIO(text)
    extra = np.loadtxt(buf)

    if extra.ndim > 1:
        extra = extra.reshape(-1)

    if extra.shape[0] != expected_len:
        raise SystemExit(
            "Length mismatch between extra data series "
            f"({extra.shape[0]}) and number of frames "
            f"({expected_len})."
        )

    return extra


def main() -> None:
    args = parse_args()

    hyperstack_path: Path = args.hyperstack

    if not hyperstack_path.is_file():
        raise SystemExit(f"Hyperstack file not found: {hyperstack_path}")

    # The directory containing the hyperstack is assumed to hold the
    # original AC TIFFs used to build it.
    tiff_dir: Path = hyperstack_path.parent

    # Load hyperstack and compute per-frame means
    stack = load_hyperstack(hyperstack_path)
    frame_means = compute_frame_means(stack)

    # Optional baseline correction using pybaselines
    frame_means = maybe_baseline_correct(
        frame_means, airpls=args.airpls_baseline, asls=args.asls_baseline
    )

    # Recover wavenumbers from original TIFF filenames using the same
    # logic as hyperstack_from_ac_tiffs.py
    wn_and_paths = collect_ac_tiffs(tiff_dir)
    wavenumbers_full = np.array([wn for wn, _ in wn_and_paths], dtype=float)

    if len(wavenumbers_full) != frame_means.shape[0]:
        raise SystemExit(
            "Mismatch between number of frames in hyperstack "
            f"({frame_means.shape[0]}) and number of AC TIFF files "
            f"with wavenumbers in the hyperstack directory ({len(wavenumbers_full)})."
        )

    # Apply wavenumber mask (edit these values to select a range)
    wn_min = 1500
    wn_max = 1800
    mask = (wavenumbers_full >= wn_min) & (wavenumbers_full <= wn_max)
    print(
        "[diagnostic] main series: "
        f"n_total={wavenumbers_full.size}, wn_range=[{wavenumbers_full.min()}, {wavenumbers_full.max()}], "
        f"mask_range=[{wn_min}, {wn_max}], n_after_mask={int(mask.sum())}"
    )
    wavenumbers = wavenumbers_full[mask]
    frame_means = frame_means[mask]

    # Prepare output path
    if args.output is not None:
        output_path = args.output
    else:
        output_path = hyperstack_path.with_name("avg_intensity_vs_wavenumber.png")

    # Plot
    plt.figure(figsize=(6, 4))
    main_label = args.legend_main if args.legend_main is not None else hyperstack_path.stem
    plt.plot(wavenumbers, frame_means, marker="o", label=main_label)

    # Optional extra series (may be provided multiple times)
    if args.extra_data:
        print(
            "[diagnostic] extra series paths: "
            + ", ".join(str(p) for p in args.extra_data)
        )

        if args.legend_extra is None:
            extra_legends = [p.stem for p in args.extra_data]
        else:
            extra_legends = list(args.legend_extra)

        print(
            "[diagnostic] extra series legends: "
            + ", ".join(repr(s) for s in extra_legends)
        )

        if len(extra_legends) != len(args.extra_data):
            raise SystemExit(
                "--legend-extra must provide exactly one legend per --extra-data. "
                f"Got {len(extra_legends)} legend(s) for {len(args.extra_data)} extra series."
            )
    else:
        extra_legends = []

    for i, extra_path in enumerate(args.extra_data):
        if not extra_path.is_file():
            raise SystemExit(f"Extra data file not found: {extra_path}")

        label = extra_legends[i] if extra_legends else extra_path.stem
        suffix = extra_path.suffix.lower()

        if suffix in {".tif", ".tiff"}:
            # Treat as a second hyperstack with its own wavenumber axis
            extra_stack = load_hyperstack(extra_path)
            extra_means = compute_frame_means(extra_stack)

            # Apply the same baseline correction to the extra series
            extra_means = maybe_baseline_correct(
                extra_means, airpls=args.airpls_baseline, asls=args.asls_baseline
            )

            # Derive wavenumbers for the extra hyperstack from AC TIFFs
            # in the same directory as that hyperstack.
            wn_and_paths_extra = collect_ac_tiffs(extra_path.parent)
            wavenumbers_extra_full = np.array(
                [wn for wn, _ in wn_and_paths_extra], dtype=float
            )

            if len(wavenumbers_extra_full) != extra_means.shape[0]:
                raise SystemExit(
                    "Mismatch between number of frames in extra "
                    "hyperstack "
                    f"({extra_means.shape[0]}) and number of AC TIFF "
                    f"files with wavenumbers in its directory "
                    f"({len(wavenumbers_extra_full)})."
                )

            extra_mask = (wavenumbers_extra_full >= wn_min) & (wavenumbers_extra_full <= wn_max)
            print(
                "[diagnostic] extra hyperstack series: "
                f"path={extra_path}, "
                f"n_total={wavenumbers_extra_full.size}, wn_range=[{wavenumbers_extra_full.min()}, {wavenumbers_extra_full.max()}], "
                f"mask_range=[{wn_min}, {wn_max}], n_after_mask={int(extra_mask.sum())}"
            )

            wavenumbers_extra = wavenumbers_extra_full[extra_mask]
            extra_means = extra_means[extra_mask]

            plt.plot(
                wavenumbers_extra,
                extra_means,
                marker="s",
                label=label,
            )
        else:
            # Treat as a text-like numeric series
            extra = load_extra_series(extra_path, expected_len=frame_means.shape[0])

            # Apply the same baseline correction to the extra series
            extra = maybe_baseline_correct(
                extra, airpls=args.airpls_baseline, asls=args.asls_baseline
            )

            plt.plot(wavenumbers, extra, marker="s", label=label)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Average intensity per frame")
    plt.title("Average intensity vs wavenumber")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    if not args.no_show:
        plt.show()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
