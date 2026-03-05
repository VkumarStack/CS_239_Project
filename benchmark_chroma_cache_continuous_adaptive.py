#!/usr/bin/env python3
import argparse
import csv
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

from benchmark_chroma_cache_pressure import (
    fetch_random_query_embeddings,
    load_collection,
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
    target_pct: float, total_used_pct: bool, vm_workers: int
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
    parser.add_argument("--initial-pressure-pct", type=float, default=0.0)
    parser.add_argument("--min-pressure-pct", type=float, default=0.0)
    parser.add_argument("--max-pressure-pct", type=float, default=95.0)
    parser.add_argument("--pressure-step-pct", type=float, default=5.0)
    parser.add_argument("--vm-workers", type=int, default=2)
    parser.add_argument("--total-used-pct", action="store_true")
    parser.add_argument("--eval-window-queries",type=int,default=200,help="Number of most-recent queries used to compute p99",)
    parser.add_argument("--min-eval-samples",type=int,default=100,help="Minimum samples required before controller adjusts pressure (prevents noisy early oscillation)",)
    parser.add_argument("--spike-threshold-ms", type=float, default=200.0, help="If p99 >= this, reduce pressure")
    parser.add_argument("--calm-threshold-ms", type=float, default=80.0, help="If p99 <= this, increase pressure")
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

    # start aggressor
    current_target = float(args.initial_pressure_pct)
    stress_proc = start_stressor_for_target(current_target, args.total_used_pct, args.vm_workers)

    start_time = time.time()
    rows: List[dict] = []
    recent: Deque[float] = deque(maxlen=args.eval_window_queries)
    qidx = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration_seconds:
                break

            emb = random.choice(query_pool)

            ef_mode = args.ef_mode
            ef_search = args.ef_search if ef_mode == "query-argument" else None

            lat_ms_list = run_queries(
                collection,
                [emb],
                top_k=10,
                ef_mode=ef_mode,
                ef_search=ef_search,
            )
            lat_ms = float(lat_ms_list[0])

            actual_pct: Optional[float] = used_mem_pct()
            vt: Optional[float] = vmtouch_residency_pct(args.path)

            rows.append(
                {
                    "query_index": qidx,
                    "elapsed_sec": round(elapsed, 6),
                    "target_pressure_pct": round(current_target, 6),
                    "actual_pressure_pct": ("" if actual_pct is None else round(actual_pct, 6)),
                    "vmtouch_residency_pct": ("" if vt is None else round(vt, 6)),
                    "latency_ms": round(lat_ms, 6),
                }
            )
            recent.append(lat_ms)
            qidx += 1

            # Evaluate controller
            if len(recent) >= min(args.min_eval_samples, args.eval_window_queries):
                p99 = pctile(list(recent), 99.0)

                changed = False
                if p99 >= args.spike_threshold_ms and current_target > args.min_pressure_pct:
                    # too high latency -> reduce pressure
                    new_target = max(args.min_pressure_pct, current_target - args.pressure_step_pct)
                    if new_target != current_target:
                        current_target = new_target
                        changed = True
                elif p99 <= args.calm_threshold_ms and current_target < args.max_pressure_pct:
                    # low latency -> increase pressure
                    new_target = min(args.max_pressure_pct, current_target + args.pressure_step_pct)
                    if new_target != current_target:
                        current_target = new_target
                        changed = True

                if changed:
                    # restart stress-ng with new target
                    stop_stress_process(stress_proc)
                    stress_proc = start_stressor_for_target(current_target, args.total_used_pct, args.vm_workers)

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
        fieldnames = ["elapsed_sec", "latency_ms", "target_pressure_pct", "actual_pressure_pct", "vmtouch_residency_pct"]
        with outp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"Wrote timeline CSV: {args.timeline_out}")


if __name__ == "__main__":
    main()
