#!/usr/bin/env python3
import argparse
import csv
import random
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Deque, List

from benchmark_chroma_cache_pressure import (
    fetch_random_query_embeddings,
    load_collection,
    run_queries,
    start_eviction_stress_vm,
    stop_stress_process,
    used_mem_pct,
    vmtouch_residency_pct,
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
        "latency_ms",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--eval-window-queries", type=int, default=200, help="Number of most-recent queries used to compute p99")
    parser.add_argument("--spike-threshold-ms", type=float, default=200.0, help="If p99 >= this, reduce pressure")
    parser.add_argument("--calm-threshold-ms", type=float, default=80.0, help="If p99 <= this, increase pressure")
    parser.add_argument("--csv-out", default="outputs/continuous_cache_adaptive.csv")
    parser.add_argument("--timeline-out", default=None)

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
    stress_proc = None
    if current_target > 0:
        stress_proc = start_eviction_stress_vm(int(current_target), None, args.vm_workers)

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
            lat_ms_list = run_queries(collection, [emb], top_k=10, ef_mode="query-argument" if False else "else", ef_search=100)
            lat_ms = lat_ms_list[0]

            actual_pct = used_mem_pct()
            vt = vmtouch_residency_pct(args.path)

            rows.append(
                {
                    "query_index": qidx,
                    "elapsed_sec": round(elapsed, 6),
                    "target_pressure_pct": round(current_target, 6),
                    "actual_pressure_pct": ("" if actual_pct is None else round(actual_pct, 6)),
                    "latency_ms": round(float(lat_ms), 6),
                }
            )
            recent.append(float(lat_ms))
            qidx += 1

            # Evaluate controller when we have enough samples
            if len(recent) >= max(10, int(args.eval_window_queries / 4)):
                p99 = statistics.quantiles(list(recent), n=100)[98]  # approximate 99th
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
                    if current_target > 0:
                        stress_proc = start_eviction_stress_vm(int(current_target), None, args.vm_workers)
                    else:
                        stress_proc = None

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
        fieldnames = ["elapsed_sec", "latency_ms", "target_pressure_pct", "actual_pressure_pct"]
        with outp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "elapsed_sec": r.get("elapsed_sec", ""),
                        "latency_ms": r.get("latency_ms", ""),
                        "target_pressure_pct": r.get("target_pressure_pct", ""),
                        "actual_pressure_pct": r.get("actual_pressure_pct", ""),
                    }
                )
        print(f"Wrote timeline CSV: {args.timeline_out}")


if __name__ == "__main__":
    main()
