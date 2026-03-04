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
        (elapsed, latency, target_pressure, actual_pressure,
         ef_search_or_None, reranked_or_None, recall_or_None)

    reranked_or_None : bool ndarray — True where rerank was performed ("yes"),
                       present only when the ``reranked`` column exists.
    recall_or_None   : float ndarray with NaN for rows where recall was not
                       computed, present only when the ``recall_at_k`` column exists.
    """
    elapsed = []
    latency = []
    target_pressure = []
    actual_pressure = []
    ef_search_vals = []
    reranked_vals = []

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

        fields = set(reader.fieldnames or [])
        has_ef_search = "ef_search" in fields
        has_reranked  = "reranked"   in fields

        for row in reader:
            elapsed.append(float(row["elapsed_sec"]))
            latency.append(float(row["latency_ms"]))
            target_pressure.append(float(row["target_pressure_pct"]))
            actual_pressure.append(float(row["actual_pressure_pct"]))
            if has_ef_search:
                ef_search_vals.append(int(row["ef_search"]))
            if has_reranked:
                v = row["reranked"].strip().lower()
                reranked_vals.append(v in ("yes", "1", "true"))

    if not elapsed:
        raise ValueError("CSV has no data rows")

    return (
        np.array(elapsed),
        np.array(latency),
        np.array(target_pressure),
        np.array(actual_pressure),
        np.array(ef_search_vals, dtype=int) if ef_search_vals else None,
        np.array(reranked_vals, dtype=bool) if reranked_vals else None,
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
    elapsed, latency, target_pressure, actual_pressure, ef_search, reranked = load_csv(csv_path)

    roll_p50 = rolling_percentile(latency, window, 50)
    roll_p99 = rolling_percentile(latency, window, 99)

    min_latency = float(np.min(latency))
    y_min = latency_y_min if latency_y_min is not None else min_latency
    if latency_y_max is not None:
        y_max = latency_y_max
    else:
        max_p99 = float(np.nanmax(roll_p99))
        min_p99 = float(np.nanmin(roll_p99))
        y_max = max(max_p99, min_p99 * 10.0)

    has_ef     = ef_search is not None
    has_rerank = reranked is not None

    # Build subplot grid: latency | [rerank status] | [ef_search] | pressure
    n_rows     = 2 + int(has_ef) + int(has_rerank)
    fig_height = 4 * n_rows

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, fig_height), sharex=True)
    # Always index axes as a list for consistent access
    axes = list(axes) if n_rows > 1 else [axes]
    ax_iter = iter(axes)

    # ── Latency subplot ────────────────────────────────────────────────────
    ax_lat = next(ax_iter)
    if not no_scatter:
        if has_rerank:
            skip_mask = ~reranked
            ax_lat.scatter(
                elapsed[reranked], latency[reranked],
                s=4, alpha=0.2, color="tab:blue", label="Latency – reranked",
            )
            ax_lat.scatter(
                elapsed[skip_mask], latency[skip_mask],
                s=4, alpha=0.3, color="tab:orange", label="Latency – rerank skipped",
            )
        else:
            ax_lat.scatter(elapsed, latency, s=4, alpha=0.2, label="Latency (raw)")

    ax_lat.plot(elapsed, roll_p50, linewidth=2, label=f"Rolling P50 (window={window})")
    ax_lat.plot(elapsed, roll_p99, linewidth=2, label=f"Rolling P99 (window={window})")

    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Query Latency Over Time")
    ax_lat.grid(alpha=0.25)
    ax_lat.legend(loc="upper left", fontsize=8)
    ax_lat.set_ylim(bottom=y_min, top=y_max)

    # ── Rerank status subplot ──────────────────────────────────────────────
    if has_rerank:
        ax_rr = next(ax_iter)
        rerank_signal = reranked.astype(int)   # 1 = reranking active, 0 = skipped
        ax_rr.step(elapsed, rerank_signal, linewidth=1.5, where="post",
                   color="tab:blue", label="Rerank active (1) / skipped (0)")
        ax_rr.fill_between(elapsed, 0, rerank_signal,
                           step="post", alpha=0.15, color="tab:blue")
        ax_rr.set_ylabel("Reranking")
        ax_rr.set_yticks([0, 1])
        ax_rr.set_yticklabels(["Skipped", "Active"])
        ax_rr.set_ylim(-0.1, 1.3)
        ax_rr.set_title("Rerank Status Over Time")
        ax_rr.grid(alpha=0.25)

    # ── ef_search subplot (adaptive benchmark only) ────────────────────────
    if has_ef:
        ax_ef = next(ax_iter)
        ax_ef.step(elapsed, ef_search, linewidth=2, where="post", label="ef_search", color="tab:purple")
        ax_ef.set_ylabel("ef_search")
        ax_ef.set_title("Adaptive ef_search Over Time")
        ax_ef.grid(alpha=0.25)
        ax_ef.legend(loc="upper left")

    # ── Memory pressure subplot ────────────────────────────────────────────
    ax_press = next(ax_iter)
    ax_press.plot(elapsed, target_pressure, linewidth=2, label="Target pressure %")
    ax_press.plot(elapsed, actual_pressure, linewidth=2, label="Actual pressure %")
    ax_press.set_xlabel("Elapsed time (s)")
    ax_press.set_ylabel("Memory pressure (%)")
    ax_press.set_title("Memory Pressure")
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
