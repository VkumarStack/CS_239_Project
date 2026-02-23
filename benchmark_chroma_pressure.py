import argparse
import csv
import inspect
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import List

import chromadb
import numpy as np


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("No values to compute percentile")
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]

    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = rank - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def load_collection(client: chromadb.PersistentClient, collection_name: str | None):
    if collection_name:
        return client.get_collection(collection_name)

    collections = client.list_collections()
    if not collections:
        raise RuntimeError("No collections found in the persisted ChromaDB path")

    return client.get_collection(collections[0].name)


def fetch_random_query_embeddings(collection, query_count: int) -> List[List[float]]:
    total = collection.count()
    if total == 0:
        raise RuntimeError("Collection is empty; cannot run query benchmark")

    if query_count > total:
        print(
            f"Requested {query_count} random queries but collection has {total} vectors. "
            f"Using {total} queries instead."
        )
        query_count = total

    random_offsets = random.sample(range(total), k=query_count)

    query_embeddings: List[List[float]] = []
    for offset in random_offsets:
        result = collection.get(limit=1, offset=offset, include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            raise RuntimeError(
                "Could not fetch stored embeddings from collection. "
                "Ensure this collection contains embeddings."
            )
        query_embeddings.append(embeddings[0])

    return query_embeddings


def apply_ef_search(collection, ef_search: int) -> str:
    current_config = collection.configuration or {}
    current_hnsw_cfg = dict(current_config.get("hnsw") or {})
    existing_ef = current_hnsw_cfg.get("ef_search")

    if existing_ef is not None and int(existing_ef) == int(ef_search):
        return "existing-configuration"

    update_error = None
    try:
        collection.modify(configuration={"hnsw": {"ef_search": int(ef_search)}})
        return "collection-configuration:hnsw.ef_search"
    except Exception as err:
        update_error = err

    query_signature = inspect.signature(collection.query)
    if "search_ef" in query_signature.parameters:
        return "query-argument"

    if existing_ef is None and int(ef_search) == 100:
        return "default-assumed"

    raise RuntimeError(
        "This ChromaDB version does not support runtime ef_search adjustment "
        "for this collection (configuration modify failed and query(search_ef=...) is unsupported). "
        f"Underlying modify error: {type(update_error).__name__}: {update_error}"
    )


def run_queries(collection, query_embeddings: List[List[float]], top_k: int, ef_mode: str, ef_search: int) -> List[float]:
    latencies_ms: List[float] = []

    for embedding in query_embeddings:
        start = time.perf_counter()
        if ef_mode == "query-argument":
            collection.query(query_embeddings=[embedding], n_results=top_k, search_ef=ef_search)
        else:
            collection.query(query_embeddings=[embedding], n_results=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    return latencies_ms


def warm_cache_with_vmtouch(data_path: str) -> None:
    if shutil.which("vmtouch") is None:
        raise RuntimeError("vmtouch not found. Install it or run without --preload-cache")

    print("Warming file cache via vmtouch...")
    cmd = ["vmtouch", "-t", "-f", "-q", data_path]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"vmtouch failed with exit code {result.returncode}")


def start_stress_process(mem_percent: int, vm_workers: int):
    if mem_percent <= 0:
        return None

    if shutil.which("stress-ng") is None:
        raise RuntimeError("stress-ng not found. Install it before running pressure ramps")

    cmd = [
        "stress-ng",
        "--vm",
        str(vm_workers),
        "--vm-bytes",
        f"{mem_percent}%",
        "--vm-keep",
        "--metrics-brief",
    ]

    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_stress_process(proc) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def parse_steps(steps_raw: str) -> List[int]:
    vals = [s.strip() for s in steps_raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("--mem-steps must have at least one value")
    steps = [int(v) for v in vals]
    for v in steps:
        if v < 0 or v > 95:
            raise ValueError("Each mem step must be in [0, 95]")
    return steps


def maybe_write_csv(csv_path: str | None, rows: List[dict]) -> None:
    if not csv_path:
        return

    fieldnames = [
        "step_index",
        "mem_percent",
        "queries",
        "query_mode",
        "top_k",
        "ef_search",
        "cache_preloaded",
        "avg_ms",
        "p50_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
    ]

    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ChromaDB query latency under ramping memory pressure (WSL-friendly)"
    )
    parser.add_argument("--path", default="chroma_data", help="Path to persisted ChromaDB directory")
    parser.add_argument("--collection", default=None, help="Collection name (default: first collection)")
    parser.add_argument("--queries-per-step", type=int, default=200, help="Queries executed at each pressure step")
    parser.add_argument("--top-k", type=int, default=10, help="n_results per query")
    parser.add_argument("--ef-search", type=int, default=100, help="HNSW ef_search value")
    parser.add_argument(
        "--mem-steps",
        default="0,20,40,60,80",
        help="Comma-separated memory pressure percentages for stress-ng vm-bytes",
    )
    parser.add_argument("--vm-workers", type=int, default=1, help="stress-ng --vm worker count")
    parser.add_argument("--settle-seconds", type=float, default=4.0, help="Wait time after starting stress before measuring")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--query-mode",
        choices=["resample", "fixed", "presampled"],
        default="resample",
        help=(
            "Query set strategy across ramp steps: "
            "resample=new random set each step, fixed=one shared set, "
            "presampled=pre-generate one set per step"
        ),
    )
    parser.add_argument("--preload-cache", action="store_true", help="Warm cache with vmtouch before test")
    parser.add_argument("--csv-out", default=None, help="Optional path to write per-step CSV results")

    args = parser.parse_args()

    if args.queries_per_step < 1:
        raise ValueError("--queries-per-step must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.ef_search < 1:
        raise ValueError("--ef-search must be >= 1")

    mem_steps = parse_steps(args.mem_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Opening persisted ChromaDB at: {args.path}")
    client = chromadb.PersistentClient(path=args.path)
    collection = load_collection(client, args.collection)
    print(f"Using collection: {collection.name}")
    print(f"Collection vector count: {collection.count()}")

    if args.preload_cache:
        warm_cache_with_vmtouch(args.path)

    ef_mode = apply_ef_search(collection, args.ef_search)
    print(f"ef_search mode: {ef_mode}")

    fixed_query_embeddings = None
    presampled_query_sets: List[List[List[float]]] = []
    if args.query_mode == "fixed":
        print("Preparing one fixed random query set from existing vectors...")
        fixed_query_embeddings = fetch_random_query_embeddings(collection, args.queries_per_step)
    elif args.query_mode == "presampled":
        print("Pre-sampling random query sets for all steps...")
        for _ in mem_steps:
            presampled_query_sets.append(fetch_random_query_embeddings(collection, args.queries_per_step))

    print("\n=== Pressure Ramp Benchmark ===")
    print(f"mem steps: {mem_steps}")
    print(f"queries/step: {args.queries_per_step}")
    print(f"query mode: {args.query_mode}")
    print(f"top_k: {args.top_k}")

    rows: List[dict] = []

    for step_idx, mem_percent in enumerate(mem_steps, start=1):
        stress_proc = None
        try:
            stress_proc = start_stress_process(mem_percent, args.vm_workers)
            if stress_proc is not None and args.settle_seconds > 0:
                time.sleep(args.settle_seconds)

            if args.query_mode == "resample":
                query_embeddings = fetch_random_query_embeddings(collection, args.queries_per_step)
            elif args.query_mode == "fixed":
                query_embeddings = fixed_query_embeddings
            else:
                query_embeddings = presampled_query_sets[step_idx - 1]

            if query_embeddings is None:
                raise RuntimeError("Query embeddings are not initialized")

            latencies_ms = run_queries(
                collection=collection,
                query_embeddings=query_embeddings,
                top_k=args.top_k,
                ef_mode=ef_mode,
                ef_search=args.ef_search,
            )

            sorted_latencies = sorted(latencies_ms)
            avg_ms = statistics.mean(sorted_latencies)
            p50_ms = percentile(sorted_latencies, 50)
            p99_ms = percentile(sorted_latencies, 99)
            min_ms = sorted_latencies[0]
            max_ms = sorted_latencies[-1]

            row = {
                "step_index": step_idx,
                "mem_percent": mem_percent,
                "queries": len(sorted_latencies),
                "query_mode": args.query_mode,
                "top_k": args.top_k,
                "ef_search": args.ef_search,
                "cache_preloaded": args.preload_cache,
                "avg_ms": round(avg_ms, 3),
                "p50_ms": round(p50_ms, 3),
                "p99_ms": round(p99_ms, 3),
                "min_ms": round(min_ms, 3),
                "max_ms": round(max_ms, 3),
            }
            rows.append(row)

            print(
                f"step {step_idx}/{len(mem_steps)} | mem={mem_percent:>2}% "
                f"| avg={avg_ms:.3f} ms | p50={p50_ms:.3f} ms | p99={p99_ms:.3f} ms"
            )
        finally:
            stop_stress_process(stress_proc)

    maybe_write_csv(args.csv_out, rows)

    if args.csv_out:
        print(f"\nWrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()