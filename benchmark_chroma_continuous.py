import argparse
import csv
import inspect
import multiprocessing as mp
import random
import shutil
import statistics
import subprocess
import sys
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


def get_mem_total_bytes() -> int:
    """Get total memory available, respecting cgroup limits if present."""
    # Try cgroups v2 first
    try:
        with open("/sys/fs/cgroup/chromabench/memory.max", "r", encoding="utf-8") as f:
            value = f.read().strip()
            if value != "max":
                return int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    
    # Try cgroups v1
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r", encoding="utf-8") as f:
            value = int(f.read().strip())
            # cgroups v1 uses a very large number to indicate "no limit"
            if value < (1 << 62):
                return value
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    
    # Fallback to system memory
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb * 1024
    raise RuntimeError("Could not read MemTotal from /proc/meminfo")


def memory_pressure_worker(
    target_bytes: object,
    actual_bytes: object,
    stop_event: object,
    chunk_bytes: int,
    pause_seconds: float,
    read_segment_pct: float = 0.0,
    read_interval_seconds: float = 0.1,
) -> None:
    chunks: List[bytearray] = []
    allocated = 0
    read_chunk_idx = 0
    read_chunk_offset = 0
    last_read_time = time.monotonic()

    while not stop_event.is_set():
        with target_bytes.get_lock():
            desired = int(target_bytes.value)

        while allocated < desired and not stop_event.is_set():
            remaining = desired - allocated
            size = chunk_bytes if remaining > chunk_bytes else remaining
            if size <= 0:
                break
            chunk = bytearray(size)
            # Touch every OS page to force physical mapping and dirty the pages,
            # preventing the kernel from using zero-page sharing or demand-paging.
            for i in range(0, size, 4096):
                chunk[i] = 0xAB
            chunks.append(chunk)
            allocated += size
            with actual_bytes.get_lock():
                actual_bytes.value = allocated

        while allocated > desired and chunks:
            removed = chunks.pop()
            allocated -= len(removed)
            # Keep the rolling-read cursor in bounds after shrinking.
            if chunks:
                read_chunk_idx = read_chunk_idx % len(chunks)
                read_chunk_offset = min(read_chunk_offset, len(chunks[read_chunk_idx]) - 1)
            else:
                read_chunk_idx = 0
                read_chunk_offset = 0
        with actual_bytes.get_lock():
            actual_bytes.value = allocated

        # Rolling read-through: scan a percentage of currently allocated memory every
        # read_interval_seconds, picking up where the previous sweep left off.  This
        # keeps pages active and creates genuine memory competition with the query workload.
        now = time.monotonic()
        if read_segment_pct > 0 and allocated > 0 and chunks and (now - last_read_time) >= read_interval_seconds:
            last_read_time = now
            bytes_remaining = max(1, int(allocated * read_segment_pct / 100.0))
            while bytes_remaining > 0:
                if read_chunk_idx >= len(chunks):
                    read_chunk_idx = 0
                    read_chunk_offset = 0
                current_chunk = chunks[read_chunk_idx]
                available = len(current_chunk) - read_chunk_offset
                to_read = min(bytes_remaining, available)
                _ = current_chunk[read_chunk_offset : read_chunk_offset + to_read]
                read_chunk_offset += to_read
                bytes_remaining -= to_read
                if read_chunk_offset >= len(current_chunk):
                    read_chunk_idx = (read_chunk_idx + 1) % len(chunks)
                    read_chunk_offset = 0

        time.sleep(pause_seconds)


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


def infer_embedding_dim(collection) -> int:
    result = collection.get(limit=1, offset=0, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        raise RuntimeError("Could not infer embedding dimension from collection")
    return len(embeddings[0])

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


def query_once(collection, embedding: List[float], top_k: int, ef_mode: str, ef_search: int) -> float:
    start = time.perf_counter()
    if ef_mode == "query-argument":
        collection.query(query_embeddings=[embedding], n_results=top_k, search_ef=ef_search)
    else:
        collection.query(query_embeddings=[embedding], n_results=top_k)
    return (time.perf_counter() - start) * 1000.0


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


# ---------------------------------------------------------------------------
# Pressure profiles — each factory returns a callable: (elapsed_sec: float) -> target_pct: float
# ---------------------------------------------------------------------------

def make_ramp_profile(start_pct: float, end_pct: float, ramp_seconds: float):
    """Linear ramp from start_pct to end_pct over ramp_seconds, then holds at end_pct."""
    def _profile(elapsed: float) -> float:
        progress = min(elapsed / ramp_seconds, 1.0)
        return start_pct + (end_pct - start_pct) * progress
    return _profile


def make_spike_profile(
    baseline_pct: float,
    peak_pct: float,
    rise_seconds: float,
    hold_seconds: float,
    fall_seconds: float,
    idle_seconds: float,
):
    """
    Periodic spike: rises linearly from baseline_pct to peak_pct over rise_seconds,
    holds at peak_pct for hold_seconds, falls linearly back to baseline_pct over
    fall_seconds, idles at baseline_pct for idle_seconds, then repeats.
    """
    period = rise_seconds + hold_seconds + fall_seconds + idle_seconds

    def _profile(elapsed: float) -> float:
        phase = elapsed % period
        if phase < rise_seconds:
            return baseline_pct + (peak_pct - baseline_pct) * (phase / rise_seconds)
        phase -= rise_seconds
        if phase < hold_seconds:
            return peak_pct
        phase -= hold_seconds
        if phase < fall_seconds:
            return peak_pct + (baseline_pct - peak_pct) * (phase / fall_seconds)
        return baseline_pct

    return _profile


def main():
    parser = argparse.ArgumentParser(
        description="Continuously ramp memory pressure with one process while continuously querying Chroma"
    )
    parser.add_argument("--path", default="chroma_data", help="Path to persisted ChromaDB directory")
    parser.add_argument("--collection", default=None, help="Collection name (default: first collection)")
    parser.add_argument("--top-k", type=int, default=10, help="n_results per query")
    parser.add_argument("--ef-search", type=int, default=100, help="HNSW ef_search value")
    parser.add_argument(
        "--query-source",
        choices=["existing", "random"],
        default="existing",
        help="Use existing stored embeddings as query vectors or generate random vectors",
    )
    parser.add_argument("--query-pool-size", type=int, default=500, help="Size of random query embedding pool")
    parser.add_argument(
        "--random-dim",
        type=int,
        default=None,
        help="Dimension for random query vectors (default: infer from collection)",
    )
    parser.add_argument("--duration-seconds", type=float, default=120.0, help="Total benchmark runtime")
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
        help="Track queries with latency >= this threshold (ms)",
    )
    parser.add_argument(
        "--max-latency-count",
        type=int,
        default=1,
        help="Abort once this many threshold breaches have occurred (requires --max-latency-ms)",
    )
    parser.add_argument("--query-interval-ms", type=float, default=0.0, help="Sleep between queries")
    parser.add_argument("--pressure-start-pct", type=float, default=0.0, help="Starting pressure percent of total RAM")
    parser.add_argument("--pressure-end-pct", type=float, default=80.0, help="Ending pressure percent of total RAM")
    parser.add_argument("--ramp-seconds", type=float, default=120.0, help="Seconds to linearly ramp pressure")
    parser.add_argument(
        "--pressure-profile",
        choices=["ramp", "spike"],
        default="ramp",
        help="Pressure shape: 'ramp' (linear, default) or 'spike' (periodic allocation pulses)",
    )
    # Spike-profile parameters (ignored when --pressure-profile=ramp)
    parser.add_argument("--spike-baseline-pct", type=float, default=0.0, help="Memory pressure %% held between spikes")
    parser.add_argument("--spike-peak-pct", type=float, default=70.0, help="Peak memory pressure %% reached during each spike")
    parser.add_argument("--spike-rise-seconds", type=float, default=5.0, help="Seconds to ramp from baseline to peak")
    parser.add_argument("--spike-hold-seconds", type=float, default=10.0, help="Seconds to hold at peak pressure")
    parser.add_argument("--spike-fall-seconds", type=float, default=5.0, help="Seconds to fall from peak back to baseline")
    parser.add_argument("--spike-idle-seconds", type=float, default=20.0, help="Seconds to idle at baseline before repeating")
    parser.add_argument("--chunk-mb", type=int, default=64, help="Allocator chunk size in MB")
    parser.add_argument("--allocator-pause-ms", type=float, default=25.0, help="Pause between allocation checks")
    parser.add_argument(
        "--read-segment-pct",
        type=float,
        default=0.0,
        help="Percentage of currently allocated memory to read per sweep cycle (0 to disable read-through)",
    )
    parser.add_argument(
        "--read-interval-ms",
        type=float,
        default=100.0,
        help="Minimum time between rolling read sweeps in milliseconds",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--csv-out", default="outputs/continuous_results.csv", help="CSV file for per-query latency timeline")
    parser.add_argument("--plot-window", type=int, default=200, help="Rolling window size for the auto-generated plot")

    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.ef_search < 1:
        raise ValueError("--ef-search must be >= 1")
    if args.query_pool_size < 1:
        raise ValueError("--query-pool-size must be >= 1")
    if args.random_dim is not None and args.random_dim < 1:
        raise ValueError("--random-dim must be >= 1")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0")
    if args.max_latency_ms is not None and args.max_latency_ms <= 0:
        raise ValueError("--max-latency-ms must be > 0")
    if args.max_latency_count < 1:
        raise ValueError("--max-latency-count must be >= 1")
    if args.max_latency_ms is None and args.max_latency_count != 1:
        raise ValueError("--max-latency-count requires --max-latency-ms")
    if args.query_interval_ms < 0:
        raise ValueError("--query-interval-ms must be >= 0")
    if args.pressure_start_pct < 0 or args.pressure_end_pct < 0:
        raise ValueError("Pressure percentages must be >= 0")
    if args.pressure_start_pct > 95 or args.pressure_end_pct > 95:
        raise ValueError("Pressure percentages must be <= 95")
    if args.ramp_seconds <= 0:
        raise ValueError("--ramp-seconds must be > 0")
    if args.chunk_mb < 1:
        raise ValueError("--chunk-mb must be >= 1")
    if args.read_segment_pct < 0 or args.read_segment_pct > 100:
        raise ValueError("--read-segment-pct must be in [0, 100]")
    if args.read_interval_ms <= 0:
        raise ValueError("--read-interval-ms must be > 0")
    if args.pressure_profile == "spike":
        if args.spike_baseline_pct < 0 or args.spike_baseline_pct > 95:
            raise ValueError("--spike-baseline-pct must be in [0, 95]")
        if args.spike_peak_pct <= args.spike_baseline_pct or args.spike_peak_pct > 95:
            raise ValueError("--spike-peak-pct must be > --spike-baseline-pct and <= 95")
        if args.spike_rise_seconds <= 0:
            raise ValueError("--spike-rise-seconds must be > 0")
        if args.spike_hold_seconds <= 0:
            raise ValueError("--spike-hold-seconds must be > 0")
        if args.spike_fall_seconds <= 0:
            raise ValueError("--spike-fall-seconds must be > 0")
        if args.spike_idle_seconds < 0:
            raise ValueError("--spike-idle-seconds must be >= 0")

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.pressure_profile == "ramp":
        profile_fn = make_ramp_profile(
            args.pressure_start_pct, args.pressure_end_pct, args.ramp_seconds
        )
    else:
        profile_fn = make_spike_profile(
            args.spike_baseline_pct, args.spike_peak_pct,
            args.spike_rise_seconds, args.spike_hold_seconds,
            args.spike_fall_seconds, args.spike_idle_seconds,
        )

    print(f"Opening persisted ChromaDB at: {args.path}")
    client = chromadb.PersistentClient(path=args.path)
    collection = load_collection(client, args.collection)
    total_vectors = collection.count()
    print(f"  Collection has {total_vectors:,} vectors")
    print(f"Using collection: {collection.name}")
    print(f"Collection vector count: {collection.count()}")

    ef_mode = apply_ef_search(collection, args.ef_search)
    print(f"ef_search mode: {ef_mode}")

    query_pool: List[List[float]] = []
    random_dim: int | None = None
    if args.query_source == "existing":
        print("Preparing query pool from existing embeddings...")
        query_pool = fetch_random_query_embeddings(collection, args.query_pool_size)
    else:
        if args.random_dim is not None:
            random_dim = args.random_dim
        else:
            random_dim = infer_embedding_dim(collection)
        print(f"Using generated random query vectors with dimension {random_dim}")

    mem_total_bytes = get_mem_total_bytes()
    print(f"MemTotal: {mem_total_bytes / (1024**3):.2f} GiB")

    target_bytes = mp.Value("Q", 0)
    actual_bytes = mp.Value("Q", 0)
    stop_event = mp.Event()

    chunk_bytes = args.chunk_mb * 1024 * 1024
    pause_seconds = args.allocator_pause_ms / 1000.0
    read_segment_pct = args.read_segment_pct
    read_interval_seconds = args.read_interval_ms / 1000.0

    pressure_proc = mp.Process(
        target=memory_pressure_worker,
        args=(
            target_bytes,
            actual_bytes,
            stop_event,
            chunk_bytes,
            pause_seconds,
            read_segment_pct,
            read_interval_seconds,
        ),
        daemon=True,
    )

    rows: List[dict] = []
    latencies: List[float] = []

    pressure_proc.start()
    start = time.perf_counter()
    next_report = 50
    stop_reason = "duration-reached"
    latency_breach_count = 0

    try:
        query_index = 0
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= args.duration_seconds:
                break

            target_pct = profile_fn(elapsed)
            target_b = int((target_pct / 100.0) * mem_total_bytes)
            with target_bytes.get_lock():
                target_bytes.value = target_b

            with actual_bytes.get_lock():
                actual_b = int(actual_bytes.value)

            if args.query_source == "existing":
                embedding = random.choice(query_pool)
            else:
                embedding = np.random.rand(random_dim).astype(np.float32).tolist()
            latency_ms = query_once(collection, embedding, args.top_k, ef_mode, args.ef_search)
            latencies.append(latency_ms)
            query_index += 1

            actual_pct = (actual_b / mem_total_bytes) * 100.0 if mem_total_bytes > 0 else 0.0

            rows.append(
                {
                    "query_index": query_index,
                    "elapsed_sec": round(elapsed, 4),
                    "target_pressure_pct": round(target_pct, 3),
                    "actual_pressure_pct": round(actual_pct, 3),
                    "latency_ms": round(latency_ms, 4),
                }
            )

            if args.max_latency_ms is not None and latency_ms >= args.max_latency_ms:
                latency_breach_count += 1
                print(
                    f"Latency breach {latency_breach_count}/{args.max_latency_count} at query {query_index}: "
                    f"{latency_ms:.3f} ms >= {args.max_latency_ms:.3f} ms"
                )
                if latency_breach_count >= args.max_latency_count:
                    stop_reason = (
                        f"max-latency-count-reached ({latency_breach_count}/{args.max_latency_count} breaches, "
                        f"threshold={args.max_latency_ms:.3f} ms)"
                    )
                    print("Safety stop triggered: breach limit reached")
                    break

            if query_index >= next_report:
                recent = latencies[max(0, len(latencies) - 50):]
                sorted_recent = sorted(recent)
                p50_recent = percentile(sorted_recent, 50)
                p99_recent = percentile(sorted_recent, 99)
                print(
                    f"q={query_index} | t={elapsed:6.1f}s | target={target_pct:5.1f}% | "
                    f"actual={actual_pct:5.1f}% | recent p50={p50_recent:.3f} ms | recent p99={p99_recent:.3f} ms"
                )
                next_report += 50

            if args.query_interval_ms > 0:
                time.sleep(args.query_interval_ms / 1000.0)
    finally:
        stop_event.set()
        pressure_proc.join(timeout=5)
        if pressure_proc.is_alive():
            pressure_proc.terminate()
            pressure_proc.join(timeout=5)

    if not latencies:
        raise RuntimeError("No queries were executed; check duration/interval settings")

    sorted_lat = sorted(latencies)
    avg = statistics.mean(sorted_lat)
    p50 = percentile(sorted_lat, 50)
    p99 = percentile(sorted_lat, 99)

    print("\n=== Continuous Ramp Results ===")
    print(f"Queries executed: {len(sorted_lat)}")
    print(f"Average: {avg:.3f} ms")
    print(f"P50: {p50:.3f} ms")
    print(f"P99: {p99:.3f} ms")
    print(f"Min: {sorted_lat[0]:.3f} ms")
    print(f"Max: {sorted_lat[-1]:.3f} ms")
    print(f"Stop reason: {stop_reason}")

    maybe_write_csv(args.csv_out, rows)
    if args.csv_out:
        print(f"Wrote CSV timeline: {args.csv_out}")
        csv_path = Path(args.csv_out)
        png_path = csv_path.with_suffix(".png")
        plot_script = Path(__file__).parent / "plot_continuous_results.py"
        try:
            subprocess.run(
                [
                    sys.executable, str(plot_script),
                    "--csv", str(csv_path),
                    "--out", str(png_path),
                    "--window", str(args.plot_window),
                    "--latency-y-max", "15",
                ],
                check=True,
            )
        except Exception as exc:
            print(f"Warning: plot generation failed: {exc}")


if __name__ == "__main__":
    main()
