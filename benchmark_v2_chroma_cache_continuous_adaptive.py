#!/usr/bin/env python3
import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

from benchmark_v2_chroma_cache_pressure import (
    fetch_random_query_embeddings,
    load_collection,
    maybe_direct_cache_transition,
    run_queries,
    start_eviction_stress_vm,
    stop_stress_process,
    used_mem_pct,
    vmtouch_residency_pct,
    mem_total_kb,
)


def maybe_write_csv(path: str | None, rows: List[dict]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_index",
        "elapsed_sec",
        "target_pressure_pct",
        "actual_pressure_pct",
        "vmtouch_residency_pct",
        "latency_ms",
        "cache_action_pre_query",
        "cache_action_post_query",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pctile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    # nearest-rank index
    k = int(round((p / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


def start_stressor_for_target(
    target_pct: float,
    total_used_pct: bool,
    vm_workers: int,
):
    if target_pct <= 0:
        return None

    if total_used_pct:
        mt_kb = mem_total_kb()
        total_bytes = int((target_pct / 100.0) * mt_kb * 1024)
        per_worker_bytes = max(1, total_bytes // max(1, vm_workers))
        return start_eviction_stress_vm(None, per_worker_bytes, vm_workers)
    else:
        return start_eviction_stress_vm(int(target_pct), None, vm_workers)


def main():
    parser = argparse.ArgumentParser(description="Adaptive cache-pressure benchmark")
    parser.add_argument("--path", default="chroma_data", help="ChromaDB persisted path")
    parser.add_argument("--collection", default=None, help="Collection name")
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--query-pool-size", type=int, default=500)
    parser.add_argument("--query-interval-ms", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--initial-pressure-pct", type=float, default=0.0)
    parser.add_argument("--min-pressure-pct", type=float, default=0.0)
    parser.add_argument("--max-pressure-pct", type=float, default=95.0)
    parser.add_argument("--pressure-step-pct", type=float, default=5.0)
    parser.add_argument("--vm-workers", type=int, default=2)
    parser.add_argument("--total-used-pct", action="store_true")
    parser.add_argument("--disable-stress", action="store_true", help="Disable stress-ng and use only direct page-cache controls.")
    parser.add_argument("--eval-window-queries",type=int,default=200,help="Number of most-recent queries used to compute p99",)
    parser.add_argument("--min-eval-samples",type=int,default=100,help="Minimum samples required before controller adjusts pressure (prevents noisy early oscillation)",)
    parser.add_argument("--controller-cooldown-queries", type=int, default=100, help="Minimum queries between controller target updates.")
    parser.add_argument("--spike-threshold-ms", type=float, default=200.0, help="If p99 >= this, reduce pressure")
    parser.add_argument("--calm-threshold-ms", type=float, default=80.0, help="If p99 <= this, increase pressure")
    parser.add_argument("--auto-latency-thresholds", action="store_true", help="Auto-calibrate calm/spike thresholds from early-query baseline.")
    parser.add_argument("--auto-threshold-warmup-queries", type=int, default=80, help="Warmup queries for auto-threshold baseline.")
    parser.add_argument("--auto-threshold-calm-multiplier", type=float, default=0.95, help="calm_threshold = baseline_p50 * multiplier")
    parser.add_argument("--auto-threshold-spike-multiplier", type=float, default=1.25, help="spike_threshold = baseline_p99 * multiplier")
    parser.add_argument("--pressure-spike-threshold-pct", type=float, default=95.0, help="If actual used memory %% >= this, reduce pressure target.")
    parser.add_argument("--pressure-calm-threshold-pct", type=float, default=75.0, help="Only increase pressure target when actual used memory %% <= this.")
    parser.add_argument("--sample-used-mem-every-n-queries", type=int, default=1, help="Sample used memory every N queries.")
    parser.add_argument("--track-vmtouch", action="store_true", help="Enable vmtouch residency sampling (expensive).")
    parser.add_argument("--sample-vmtouch-every-n-queries", type=int, default=50, help="If --track-vmtouch, sample vmtouch every N queries.")
    parser.add_argument(
        "--direct-cache-mode",
        choices=["none", "evict", "willneed", "readwarm"],
        default="none",
        help="Direct page-cache transition mode.",
    )
    parser.add_argument(
        "--direct-cache-min-size-bytes",
        type=int,
        default=4096,
        help="Ignore tiny files when applying direct cache transitions.",
    )
    parser.add_argument(
        "--direct-cache-on-start",
        action="store_true",
        help="Apply direct cache transition once before adaptive loop starts.",
    )
    parser.add_argument(
        "--direct-cache-on-change",
        action="store_true",
        help="Apply direct cache transition whenever controller changes target pressure.",
    )
    parser.add_argument(
        "--direct-cache-every-n-queries",
        type=int,
        default=0,
        help="If >0, apply direct cache transition every N queries.",
    )
    parser.add_argument("--direct-cache-verbose", action="store_true")
    parser.add_argument("--csv-out", default="outputs/continuous_cache_adaptive.csv")
    parser.add_argument("--timeline-out", default=None)
    parser.add_argument("--ef-mode", choices=["query-argument", "collection-default"], default="collection-default", help="How to set ef_search in queries",)
    parser.add_argument("--ef-search", type=int, default=100)

    args = parser.parse_args()

    print(f"Opening persisted ChromaDB at: {args.path}")
    import chromadb

    client = chromadb.PersistentClient(path=args.path)
    collection = load_collection(client, args.collection)
    print(f"Using collection: {collection.name} ({collection.count()} vectors)")

    # Prepare query pool
    query_pool = fetch_random_query_embeddings(collection, args.query_pool_size)
    print(f"Prepared query pool with {len(query_pool)} generated embeddings")

    # Start aggressor.
    current_target = float(args.initial_pressure_pct)
    stress_proc = None
    if not args.disable_stress:
        stress_proc = start_stressor_for_target(current_target, args.total_used_pct, args.vm_workers)

    if args.direct_cache_on_start and args.direct_cache_mode != "none":
        maybe_direct_cache_transition(
            data_path=args.path,
            mode=args.direct_cache_mode,
            min_size_bytes=args.direct_cache_min_size_bytes,
            verbose=args.direct_cache_verbose,
        )
    pending_pre_query_action = "on_start" if (args.direct_cache_on_start and args.direct_cache_mode != "none") else ""

    start_time = time.time()
    rows: List[dict] = []
    recent: Deque[float] = deque(maxlen=args.eval_window_queries)
    warmup_latencies: List[float] = []
    thresholds_auto_set = False
    last_adjust_qidx = -10**9
    current_spike_threshold = float(args.spike_threshold_ms)
    current_calm_threshold = float(args.calm_threshold_ms)
    last_actual_pct: Optional[float] = None
    last_vt: Optional[float] = None
    qidx = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration_seconds:
                break

            emb = random.choice(query_pool)
            pre_query_action = pending_pre_query_action
            pending_pre_query_action = ""
            if (
                args.direct_cache_mode != "none"
                and args.direct_cache_every_n_queries > 0
                and qidx > 0
                and (qidx % args.direct_cache_every_n_queries == 0)
            ):
                maybe_direct_cache_transition(
                    data_path=args.path,
                    mode=args.direct_cache_mode,
                    min_size_bytes=args.direct_cache_min_size_bytes,
                    verbose=args.direct_cache_verbose,
                )
                pre_query_action = f"periodic:{args.direct_cache_mode}"

            ef_mode = args.ef_mode
            ef_search = args.ef_search if ef_mode == "query-argument" else None

            lat_ms_list = run_queries(
                collection,
                [emb],
                top_k=args.top_k,
                ef_mode=ef_mode,
                ef_search=ef_search,
            )
            lat_ms = float(lat_ms_list[0])

            if args.sample_used_mem_every_n_queries > 0 and (qidx % args.sample_used_mem_every_n_queries == 0):
                last_actual_pct = used_mem_pct()

            if (
                args.track_vmtouch
                and args.sample_vmtouch_every_n_queries > 0
                and (qidx % args.sample_vmtouch_every_n_queries == 0)
            ):
                last_vt = vmtouch_residency_pct(args.path)

            actual_pct = last_actual_pct
            vt = last_vt

            rows.append(
                {
                    "query_index": qidx,
                    "elapsed_sec": round(elapsed, 6),
                    "target_pressure_pct": round(current_target, 6),
                    "actual_pressure_pct": ("" if actual_pct is None else round(actual_pct, 6)),
                    "vmtouch_residency_pct": ("" if vt is None else round(vt, 6)),
                    "latency_ms": round(lat_ms, 6),
                    "cache_action_pre_query": pre_query_action,
                    "cache_action_post_query": "",
                }
            )
            recent.append(lat_ms)
            qidx += 1

            if args.auto_latency_thresholds and not thresholds_auto_set:
                warmup_latencies.append(lat_ms)
                if len(warmup_latencies) >= max(10, args.auto_threshold_warmup_queries):
                    baseline_p50 = pctile(warmup_latencies, 50.0)
                    baseline_p99 = pctile(warmup_latencies, 99.0)
                    current_calm_threshold = max(0.001, baseline_p50 * args.auto_threshold_calm_multiplier)
                    current_spike_threshold = max(current_calm_threshold + 0.001, baseline_p99 * args.auto_threshold_spike_multiplier)
                    thresholds_auto_set = True
                    print(
                        "Auto thresholds set: "
                        f"calm={current_calm_threshold:.3f}ms, spike={current_spike_threshold:.3f}ms "
                        f"(from baseline p50={baseline_p50:.3f}, p99={baseline_p99:.3f})"
                    )

            # Evaluate controller
            if len(recent) >= min(args.min_eval_samples, args.eval_window_queries):
                if qidx - last_adjust_qidx < max(0, args.controller_cooldown_queries):
                    if args.query_interval_ms > 0:
                        time.sleep(args.query_interval_ms / 1000.0)
                    continue

                p99 = pctile(list(recent), 99.0)

                changed = False
                if actual_pct is not None and actual_pct >= args.pressure_spike_threshold_pct and current_target > args.min_pressure_pct:
                    # memory pressure spike -> reduce target
                    new_target = max(args.min_pressure_pct, current_target - args.pressure_step_pct)
                    if new_target != current_target:
                        current_target = new_target
                        changed = True
                elif p99 >= current_spike_threshold and current_target > args.min_pressure_pct:
                    # too high latency -> reduce pressure
                    new_target = max(args.min_pressure_pct, current_target - args.pressure_step_pct)
                    if new_target != current_target:
                        current_target = new_target
                        changed = True
                elif p99 <= current_calm_threshold and current_target < args.max_pressure_pct:
                    # low latency + calm memory pressure -> increase pressure
                    if actual_pct is None or actual_pct <= args.pressure_calm_threshold_pct:
                        new_target = min(args.max_pressure_pct, current_target + args.pressure_step_pct)
                        if new_target != current_target:
                            current_target = new_target
                            changed = True

                if changed:
                    # Restart stress-ng with new target, unless stress is disabled.
                    if not args.disable_stress:
                        stop_stress_process(stress_proc)
                        stress_proc = start_stressor_for_target(current_target, args.total_used_pct, args.vm_workers)
                    if args.direct_cache_on_change and args.direct_cache_mode != "none":
                        maybe_direct_cache_transition(
                            data_path=args.path,
                            mode=args.direct_cache_mode,
                            min_size_bytes=args.direct_cache_min_size_bytes,
                            verbose=args.direct_cache_verbose,
                        )
                        if rows:
                            rows[-1]["cache_action_post_query"] = f"on_change:{args.direct_cache_mode}"
                    last_adjust_qidx = qidx

            # sleep between queries
            if args.query_interval_ms > 0:
                time.sleep(args.query_interval_ms / 1000.0)

    finally:
        stop_stress_process(stress_proc)

    # Write outputs
    if args.csv_out:
        maybe_write_csv(args.csv_out, rows)
        print(f"Wrote CSV: {args.csv_out}")

    if args.timeline_out:
        # timeline schema matches plotting tool
        outp = Path(args.timeline_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "elapsed_sec",
            "latency_ms",
            "target_pressure_pct",
            "actual_pressure_pct",
            "vmtouch_residency_pct",
            "cache_action_pre_query",
            "cache_action_post_query",
        ]
        with outp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"Wrote timeline CSV: {args.timeline_out}")


if __name__ == "__main__":
    main()
