import argparse
import csv
from pathlib import Path

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
    elapsed = []
    latency = []
    target_pressure = []
    actual_pressure = []

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

        for row in reader:
            elapsed.append(float(row["elapsed_sec"]))
            latency.append(float(row["latency_ms"]))
            target_pressure.append(float(row["target_pressure_pct"]))
            actual_pressure.append(float(row["actual_pressure_pct"]))

    if not elapsed:
        raise ValueError("CSV has no data rows")

    return (
        np.array(elapsed),
        np.array(latency),
        np.array(target_pressure),
        np.array(actual_pressure),
    )


def main():
    parser = argparse.ArgumentParser(description="Plot Chroma continuous benchmark CSV")
    parser.add_argument("--csv", default="continuous_results.csv", help="Input CSV file")
    parser.add_argument("--out", default="continuous_results_plot.png", help="Output PNG path")
    parser.add_argument("--window", type=int, default=200, help="Rolling window size in number of queries")
    parser.add_argument("--latency-y-min", type=float, default=None, help="Optional latency subplot y-axis minimum")
    parser.add_argument("--latency-y-max", type=float, default=None, help="Optional latency subplot y-axis maximum")
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

    elapsed, latency, target_pressure, actual_pressure = load_csv(csv_path)

    roll_p50 = rolling_percentile(latency, args.window, 50)
    roll_p99 = rolling_percentile(latency, args.window, 99)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_lat = axes[0]
    if not args.no_scatter:
        ax_lat.scatter(elapsed, latency, s=4, alpha=0.2, label="Latency (raw)")
    ax_lat.plot(elapsed, roll_p50, linewidth=2, label=f"Rolling P50 (window={args.window})")
    ax_lat.plot(elapsed, roll_p99, linewidth=2, label=f"Rolling P99 (window={args.window})")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Chroma Query Latency Over Time")
    ax_lat.grid(alpha=0.25)
    ax_lat.legend(loc="upper left")
    if args.latency_y_min is not None or args.latency_y_max is not None:
        ax_lat.set_ylim(bottom=args.latency_y_min, top=args.latency_y_max)

    ax_press = axes[1]
    ax_press.plot(elapsed, target_pressure, linewidth=2, label="Target pressure %")
    ax_press.plot(elapsed, actual_pressure, linewidth=2, label="Actual pressure %")
    ax_press.set_xlabel("Elapsed time (s)")
    ax_press.set_ylabel("Memory pressure (%)")
    ax_press.set_title("Memory Pressure Ramp")
    ax_press.grid(alpha=0.25)
    ax_press.legend(loc="upper left")
    if args.pressure_y_min is not None or args.pressure_y_max is not None:
        ax_press.set_ylim(bottom=args.pressure_y_min, top=args.pressure_y_max)

    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
