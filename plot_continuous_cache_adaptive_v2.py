#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def rolling_percentile(values: np.ndarray, window: int, pct: float) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be >= 1")
    out = np.full(values.shape, np.nan, dtype=float)
    for i in range(window - 1, len(values)):
        out[i] = np.percentile(values[i - window + 1 : i + 1], pct)
    return out


def rolling_qps(elapsed_sec: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        window = 2
    out = np.full(elapsed_sec.shape, np.nan, dtype=float)
    for i in range(window - 1, len(elapsed_sec)):
        dt = elapsed_sec[i] - elapsed_sec[i - window + 1]
        out[i] = (window - 1) / dt if dt > 0 else np.nan
    return out


def _parse_float(value: str) -> float:
    s = (value or "").strip()
    if s == "":
        return np.nan
    return float(s)


def load_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    elapsed = []
    latency = []
    target_pressure = []
    actual_pressure = []
    vmtouch_res = []
    query_index = []
    cache_action_pre = []
    cache_action_post = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"elapsed_sec", "latency_ms", "target_pressure_pct", "actual_pressure_pct"}
        if not required.issubset(fields):
            raise ValueError(
                "CSV missing required columns. Expected: "
                "elapsed_sec, latency_ms, target_pressure_pct, actual_pressure_pct"
            )

        has_query_index = "query_index" in fields
        has_vmtouch = "vmtouch_residency_pct" in fields
        has_cache_action_pre = "cache_action_pre_query" in fields
        has_cache_action_post = "cache_action_post_query" in fields

        for row in reader:
            elapsed.append(float(row["elapsed_sec"]))
            latency.append(float(row["latency_ms"]))
            target_pressure.append(_parse_float(row["target_pressure_pct"]))
            actual_pressure.append(_parse_float(row["actual_pressure_pct"]))
            if has_vmtouch:
                vmtouch_res.append(_parse_float(row["vmtouch_residency_pct"]))
            if has_query_index:
                query_index.append(int(float(row["query_index"])))
            if has_cache_action_pre:
                cache_action_pre.append((row.get("cache_action_pre_query") or "").strip())
            if has_cache_action_post:
                cache_action_post.append((row.get("cache_action_post_query") or "").strip())

    if not elapsed:
        raise ValueError("CSV has no data rows")

    if not has_query_index:
        query_index = list(range(len(elapsed)))
    if not has_vmtouch:
        vmtouch_res = [np.nan] * len(elapsed)
    if not has_cache_action_pre:
        cache_action_pre = [""] * len(elapsed)
    if not has_cache_action_post:
        cache_action_post = [""] * len(elapsed)

    return {
        "elapsed_sec": np.array(elapsed, dtype=float),
        "latency_ms": np.array(latency, dtype=float),
        "target_pressure_pct": np.array(target_pressure, dtype=float),
        "actual_pressure_pct": np.array(actual_pressure, dtype=float),
        "vmtouch_residency_pct": np.array(vmtouch_res, dtype=float),
        "query_index": np.array(query_index, dtype=int),
        "cache_action_pre_query": np.array(cache_action_pre, dtype=str),
        "cache_action_post_query": np.array(cache_action_post, dtype=str),
    }


def make_plot(
    csv_path: Path,
    out_path: Path,
    latency_window: int,
    qps_window: int,
    show_scatter: bool,
    pressure_y_min: Optional[float],
    pressure_y_max: Optional[float],
) -> None:
    d = load_csv(csv_path)
    elapsed = d["elapsed_sec"]
    latency = d["latency_ms"]
    target = d["target_pressure_pct"]
    actual = d["actual_pressure_pct"]
    vmtouch = d["vmtouch_residency_pct"]
    cache_pre = d["cache_action_pre_query"]
    cache_post = d["cache_action_post_query"]

    lat_p50 = rolling_percentile(latency, latency_window, 50.0)
    lat_p95 = rolling_percentile(latency, latency_window, 95.0)
    lat_p99 = rolling_percentile(latency, latency_window, 99.0)
    qps = rolling_qps(elapsed, qps_window)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    ax_lat, ax_press, ax_qps = axes

    if show_scatter:
        ax_lat.scatter(elapsed, latency, s=6, alpha=0.22, label="Latency (raw)")
    ax_lat.plot(elapsed, lat_p50, linewidth=2.0, label=f"Rolling P50 ({latency_window})")
    ax_lat.plot(elapsed, lat_p95, linewidth=2.0, label=f"Rolling P95 ({latency_window})")
    ax_lat.plot(elapsed, lat_p99, linewidth=2.0, label=f"Rolling P99 ({latency_window})")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("Adaptive Cache Benchmark: Latency Dynamics")
    ax_lat.grid(alpha=0.25)
    ax_lat.legend(loc="upper left", fontsize=9)

    ax_press.step(elapsed, target, where="post", linewidth=2.0, label="Target pressure %")
    ax_press.plot(elapsed, actual, linewidth=1.8, label="Actual pressure %")
    if np.any(~np.isnan(vmtouch)):
        ax_press.plot(elapsed, vmtouch, linewidth=1.4, alpha=0.9, label="vmtouch residency %")
    change_mask = np.r_[False, np.diff(target) != 0]
    if np.any(change_mask):
        ax_press.scatter(elapsed[change_mask], target[change_mask], s=24, marker="x", label="Controller updates")
    pre_mask = cache_pre != ""
    post_mask = cache_post != ""
    if np.any(pre_mask):
        ax_press.scatter(
            elapsed[pre_mask],
            target[pre_mask],
            s=26,
            marker="o",
            facecolors="none",
            linewidths=1.2,
            label="Cache action (pre-query)",
        )
    if np.any(post_mask):
        ax_press.scatter(
            elapsed[post_mask],
            target[post_mask],
            s=28,
            marker="^",
            label="Cache action (post-query)",
        )
    ax_press.set_ylabel("Pressure (%)")
    ax_press.set_title("Controller and Pressure Signals")
    if pressure_y_min is not None or pressure_y_max is not None:
        ax_press.set_ylim(bottom=pressure_y_min, top=pressure_y_max)
    ax_press.grid(alpha=0.25)
    ax_press.legend(loc="upper left", fontsize=9)

    ax_qps.plot(elapsed, qps, linewidth=2.0, color="tab:green", label=f"Rolling QPS ({qps_window})")
    ax_qps.set_xlabel("Elapsed time (s)")
    ax_qps.set_ylabel("Queries/sec")
    ax_qps.set_title("Throughput Over Time")
    ax_qps.grid(alpha=0.25)
    ax_qps.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot v2 adaptive cache benchmark results")
    parser.add_argument("--csv", default="outputs/continuous_cache_adaptive_v2.csv", help="Input CSV path")
    parser.add_argument("--out", default=None, help="Output image path (default: CSV with .png)")
    parser.add_argument("--latency-window", type=int, default=150, help="Rolling latency window (queries)")
    parser.add_argument("--qps-window", type=int, default=120, help="Rolling QPS window (queries)")
    parser.add_argument("--no-scatter", action="store_true", help="Disable raw latency scatter")
    parser.add_argument("--pressure-y-min", type=float, default=None, help="Optional pressure axis min")
    parser.add_argument("--pressure-y-max", type=float, default=None, help="Optional pressure axis max")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")

    make_plot(
        csv_path=csv_path,
        out_path=out_path,
        latency_window=args.latency_window,
        qps_window=args.qps_window,
        show_scatter=not args.no_scatter,
        pressure_y_min=args.pressure_y_min,
        pressure_y_max=args.pressure_y_max,
    )


if __name__ == "__main__":
    main()
