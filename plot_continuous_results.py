import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def rolling_percentile(values: np.ndarray, window: int, pct: float) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be >= 1")

    out = np.full(values.shape, np.nan, dtype=float)
    for i in range(window - 1, len(values)):
        window_slice = values[i - window + 1 : i + 1]
        out[i] = np.percentile(window_slice, pct)
    return out


def load_csv(csv_path: Path):
    """Load benchmark CSV.

    Returns a tuple of:
        (elapsed, latency, target_pressure, actual_pressure, ef_search_or_None)
    where ef_search_or_None is a numpy int array when the column is present, else None.
    """
    elapsed = []
    latency = []
    target_pressure = []
    actual_pressure = []
    ef_search_vals = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "elapsed_sec",
            "latency_ms",
            "target_pressure_pct",
            "actual_pressure_pct",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "CSV missing required columns. Expected: "
                "elapsed_sec, latency_ms, target_pressure_pct, actual_pressure_pct"
            )

        has_ef_search = "ef_search" in (reader.fieldnames or [])

        for row in reader:
            elapsed.append(float(row["elapsed_sec"]))
            latency.append(float(row["latency_ms"]))
            target_pressure.append(float(row["target_pressure_pct"]))
            actual_pressure.append(float(row["actual_pressure_pct"]))
            if has_ef_search:
                ef_search_vals.append(int(row["ef_search"]))

    if not elapsed:
        raise ValueError("CSV has no data rows")

    return (
        np.array(elapsed),
        np.array(latency),
        np.array(target_pressure),
        np.array(actual_pressure),
        np.array(ef_search_vals, dtype=int) if ef_search_vals else None,
    )


def make_plot(
    csv_path: Path,
    out_path: Path,
    window: int = 200,
    latency_y_min: Optional[float] = None,
    latency_y_max: Optional[float] = None,
    pressure_y_min: Optional[float] = None,
    pressure_y_max: Optional[float] = None,
    no_scatter: bool = False,
) -> None:
    """Generate and save the benchmark plot."""
    elapsed, latency, target_pressure, actual_pressure, ef_search = load_csv(csv_path)

    roll_p50 = rolling_percentile(latency, window, 50)
    roll_p99 = rolling_percentile(latency, window, 99)

    # Auto-compute latency y-axis limits from the data when not explicitly provided.
    # Bottom: the minimum observed latency; Top: at most 20x the minimum latency.
    min_latency = float(np.min(latency))
    y_min = latency_y_min if latency_y_min is not None else min_latency
    y_max = latency_y_max if latency_y_max is not None else min_latency * 20.0

    has_ef = ef_search is not None
    n_rows = 3 if has_ef else 2
    fig_height = 12 if has_ef else 8

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_height), sharex=True)

    ax_lat = axes[0]
    if not no_scatter:
        ax_lat.scatter(elapsed, latency, s=4, alpha=0.2, label="Latency (raw)")
    ax_lat.plot(elapsed, roll_p50, linewidth=2, label=f"Rolling P50 (window={window})")
    ax_lat.plot(elapsed, roll_p99, linewidth=2, label=f"Rolling P99 (window={window})")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Chroma Query Latency Over Time")
    ax_lat.grid(alpha=0.25)
    ax_lat.legend(loc="upper left")
    ax_lat.set_ylim(bottom=y_min, top=y_max)

    if has_ef:
        ax_ef = axes[1]
        ax_ef.step(elapsed, ef_search, linewidth=2, where="post", label="ef_search", color="tab:purple")
        ax_ef.set_ylabel("ef_search")
        ax_ef.set_title("Adaptive ef_search Over Time")
        ax_ef.grid(alpha=0.25)
        ax_ef.legend(loc="upper left")
        ax_press = axes[2]
    else:
        ax_press = axes[1]

    ax_press.plot(elapsed, target_pressure, linewidth=2, label="Target pressure %")
    ax_press.plot(elapsed, actual_pressure, linewidth=2, label="Actual pressure %")
    ax_press.set_xlabel("Elapsed time (s)")
    ax_press.set_ylabel("Memory pressure (%)")
    ax_press.set_title("Memory Pressure Ramp")
    ax_press.grid(alpha=0.25)
    ax_press.legend(loc="upper left")
    if pressure_y_min is not None or pressure_y_max is not None:
        ax_press.set_ylim(bottom=pressure_y_min, top=pressure_y_max)

    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Chroma continuous benchmark CSV")
    parser.add_argument("--csv", default="outputs/continuous_results.csv", help="Input CSV file")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same as CSV with .png extension)")
    parser.add_argument("--window", type=int, default=200, help="Rolling window size in number of queries")
    parser.add_argument("--latency-y-min", type=float, default=None, help="Override latency subplot y-axis minimum (default: data minimum)")
    parser.add_argument("--latency-y-max", type=float, default=None, help="Override latency subplot y-axis maximum (default: 20x data minimum)")
    parser.add_argument("--pressure-y-min", type=float, default=None, help="Optional pressure subplot y-axis minimum")
    parser.add_argument("--pressure-y-max", type=float, default=None, help="Optional pressure subplot y-axis maximum")
    parser.add_argument(
        "--no-scatter",
        action="store_true",
        help="Disable raw latency scatter points",
    )

    args = parser.parse_args()

    if (
        args.latency_y_min is not None
        and args.latency_y_max is not None
        and args.latency_y_min >= args.latency_y_max
    ):
        raise ValueError("--latency-y-min must be less than --latency-y-max")

    if (
        args.pressure_y_min is not None
        and args.pressure_y_max is not None
        and args.pressure_y_min >= args.pressure_y_max
    ):
        raise ValueError("--pressure-y-min must be less than --pressure-y-max")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path = Path(args.out) if args.out is not None else csv_path.with_suffix(".png")

    make_plot(
        csv_path=csv_path,
        out_path=out_path,
        window=args.window,
        latency_y_min=args.latency_y_min,
        latency_y_max=args.latency_y_max,
        pressure_y_min=args.pressure_y_min,
        pressure_y_max=args.pressure_y_max,
        no_scatter=args.no_scatter,
    )


if __name__ == "__main__":
    main()
