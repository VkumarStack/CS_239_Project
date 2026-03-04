"""benchmark_faiss_baseline.py

Single-pass FAISS baseline benchmark for comparison with benchmark_faiss_twophase.py.

Supports two index types (selected via --index-type):

  int8      IndexHNSWSQ (QT_8bit) — quantized HNSW, no reranking.
            Pass-1 only, identical to benchmark_faiss_twophase.py with
            --rerank-threshold-pct 0.  Provided here as a convenience so
            you don't have to remember the right flags.

  float32   IndexHNSWFlat — full-precision HNSW.
            Stores float32 vectors natively inside the index; no separate
            rerank store.  Ground-truth latency baseline.

Memory budget formulae
----------------------
  float32 HNSW  : N × (4×dim + 8×M) bytes
  int8 HNSW     : N × (  dim + 8×M) bytes

Sharing data with the two-phase benchmark
-----------------------------------------
Point --shared-data-dir at the two-phase cache directory and this script
will reuse the same vectors.npy and queries.npy, only rebuilding the FAISS
index.  This lets you compare all three approaches on an identical dataset.

  # Build two-phase cache first (or reuse existing)
  python benchmark_faiss_twophase.py --total-index-gb 0.5 --dim 128 --duration-seconds 1

  # Float32 baseline on the same dataset
  python benchmark_faiss_baseline.py --index-type float32 --shared-data-dir index_cache \\
      --total-index-gb 0.5 --dim 128 \\
      --pressure-profile spike --spike-peak-pct 80 --duration-seconds 100

  # Int8-only on the same dataset
  python benchmark_faiss_baseline.py --index-type int8 --shared-data-dir index_cache \\
      --total-index-gb 0.5 --dim 128 \\
      --pressure-profile spike --spike-peak-pct 80 --duration-seconds 100
"""
import argparse
import csv
import json
import multiprocessing as mp
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Utilities (identical to benchmark_faiss_twophase.py)
# ---------------------------------------------------------------------------

def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("No values")
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    w = rank - lower
    return sorted_values[lower] * (1 - w) + sorted_values[upper] * w


def get_mem_total_bytes() -> int:
    for cg_path in (
        "/sys/fs/cgroup/chromabench/memory.max",
    ):
        try:
            with open(cg_path, encoding="utf-8") as f:
                v = f.read().strip()
                if v != "max":
                    return int(v)
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", encoding="utf-8") as f:
            v = int(f.read().strip())
            if v < (1 << 62):
                return v
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("Could not read MemTotal from /proc/meminfo")


def memory_pressure_worker(
    target_bytes, actual_bytes, stop_event,
    chunk_bytes, pause_seconds,
    read_segment_pct=0.0, read_interval_seconds=0.1,
):
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
            for i in range(0, size, 4096):
                chunk[i] = 0xAB
            chunks.append(chunk)
            allocated += size
            with actual_bytes.get_lock():
                actual_bytes.value = allocated

        while allocated > desired and chunks:
            removed = chunks.pop()
            allocated -= len(removed)
            if chunks:
                read_chunk_idx = read_chunk_idx % len(chunks)
                read_chunk_offset = min(read_chunk_offset, len(chunks[read_chunk_idx]) - 1)
            else:
                read_chunk_idx = 0
                read_chunk_offset = 0
        with actual_bytes.get_lock():
            actual_bytes.value = allocated

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
                _ = current_chunk[read_chunk_offset: read_chunk_offset + to_read]
                read_chunk_offset += to_read
                bytes_remaining -= to_read
                if read_chunk_offset >= len(current_chunk):
                    read_chunk_idx = (read_chunk_idx + 1) % len(chunks)
                    read_chunk_offset = 0

        time.sleep(pause_seconds)


# ---------------------------------------------------------------------------
# Memory-budget sizing
# ---------------------------------------------------------------------------

def compute_num_vectors(total_gb: float, dim: int, m: int, index_type: str) -> int:
    """Compute N so the index fits within *total_gb* GiB.

    float32 HNSW : N × (4×dim + 8×M)  bytes
    int8 HNSW    : N × (  dim + 8×M)  bytes
    """
    total_bytes = total_gb * (1024 ** 3)
    if index_type == "float32":
        bytes_per_vector = 4 * dim + 8 * m
    else:  # int8
        bytes_per_vector = dim + 8 * m
    return max(1000, int(total_bytes / bytes_per_vector))


def print_memory_budget(n: int, dim: int, m: int, index_type: str) -> None:
    if index_type == "float32":
        vec_gb   = n * dim * 4 / (1024 ** 3)
        vec_label = "float32 vectors"
    else:
        vec_gb   = n * dim * 1 / (1024 ** 3)
        vec_label = "int8 SQ vectors "
    graph_gb = n * 2 * m * 4 / (1024 ** 3)
    total_gb = vec_gb + graph_gb
    print(f"\n{'─'*52}")
    print(f"  Index memory budget  [{index_type} HNSW]")
    print(f"{'─'*52}")
    print(f"  Vectors (N)     : {n:>12,}")
    print(f"  Dimension       : {dim:>12,}")
    print(f"  HNSW M          : {m:>12,}")
    print(f"  {vec_label} : {vec_gb:>11.3f} GiB")
    print(f"  HNSW graph      : {graph_gb:>11.3f} GiB")
    print(f"  ─────────────────────────────────────")
    print(f"  Total (approx.) : {total_gb:>11.3f} GiB")
    print(f"{'─'*52}\n")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_index(
    data: np.ndarray,
    index_type: str,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> faiss.Index:
    n, dim = data.shape
    if index_type == "float32":
        print(f"Building IndexHNSWFlat (float32, M={m}, ef_construction={ef_construction}) on {n:,} vectors…")
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
    else:
        print(f"Building IndexHNSWSQ (QT_8bit, M={m}, ef_construction={ef_construction}) on {n:,} vectors…")
        index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, m, faiss.METRIC_L2)
        train_n = min(n, 50_000)
        print(f"  Training scalar quantizer on {train_n:,} vectors…")
        index.train(data[:train_n])

    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    print(f"  Adding {n:,} vectors…")
    batch = 100_000
    for start in range(0, n, batch):
        index.add(data[start: start + batch])
        if start > 0 and start % 500_000 == 0:
            print(f"    added {start:,} / {n:,}")

    print(f"  Index built: {index.ntotal:,} vectors stored")
    return index


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_meta(n: int, dim: int, m: int, ef_construction: int, seed: int, index_type: str) -> dict:
    return {
        "n_vectors": n, "dim": dim, "m": m,
        "ef_construction": ef_construction, "seed": seed,
        "index_type": index_type,
    }


def _cache_paths(cache_dir: Path):
    return (
        cache_dir / "index.faiss",
        cache_dir / "vectors.npy",
        cache_dir / "queries.npy",
        cache_dir / "meta.json",
    )


def try_load_cache(
    cache_dir: Path,
    n: int, dim: int, m: int, ef_construction: int, seed: int,
    ef_search: int, index_type: str,
) -> Optional[tuple]:
    idx_path, vec_path, qry_path, meta_path = _cache_paths(cache_dir)
    if not all(p.exists() for p in (idx_path, vec_path, qry_path, meta_path)):
        return None
    try:
        with meta_path.open() as f:
            saved = json.load(f)
    except Exception:
        return None
    if saved != _cache_meta(n, dim, m, ef_construction, seed, index_type):
        print("Cache found but parameters differ — rebuilding.")
        return None
    print(f"Loading cached index from: {cache_dir}")
    index = faiss.read_index(str(idx_path))
    index.hnsw.efSearch = ef_search
    # float32 vectors mmap'd for the int8 case (not used for search, only here
    # so the file is present; float32 case stores vectors inside the index).
    vectors = np.load(str(vec_path), mmap_mode="r")
    queries = np.load(str(qry_path))
    print(f"  Loaded {index.ntotal:,} vectors")
    return index, vectors, queries


def save_cache(
    cache_dir: Path,
    index: faiss.Index,
    vectors: np.ndarray,
    queries: np.ndarray,
    n: int, dim: int, m: int, ef_construction: int, seed: int, index_type: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx_path, vec_path, qry_path, meta_path = _cache_paths(cache_dir)
    print(f"Saving index cache to: {cache_dir}")
    faiss.write_index(index, str(idx_path))
    np.save(str(vec_path), vectors)
    np.save(str(qry_path), queries)
    with meta_path.open("w") as fh:
        json.dump(_cache_meta(n, dim, m, ef_construction, seed, index_type), fh, indent=2)
    print(f"  Saved {index.ntotal:,} vectors")


def try_load_shared_data(
    shared_dir: Path, dim: int, seed: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load vectors.npy and queries.npy from a two-phase cache directory.

    Only validates dim and seed — N is intentionally not checked because the
    two-phase benchmark sizes N using a different memory formula, so the stored
    N will never match what this script would compute independently.
    After loading, the caller should derive n_vectors from vectors.shape[0].
    """
    vec_path = shared_dir / "vectors.npy"
    qry_path = shared_dir / "queries.npy"
    meta_path = shared_dir / "meta.json"
    if not all(p.exists() for p in (vec_path, qry_path, meta_path)):
        return None
    try:
        with meta_path.open() as f:
            saved = json.load(f)
    except Exception:
        return None
    if saved.get("dim") != dim or saved.get("seed") != seed:
        print(
            f"Shared data dir has dim={saved.get('dim')}, seed={saved.get('seed')} "
            f"— does not match requested dim={dim}, seed={seed}. "
            f"Ignoring --shared-data-dir."
        )
        return None
    print(f"Reusing vectors and queries from shared data dir: {shared_dir}")
    print(f"  Shared N={saved.get('n_vectors'):,} (overrides computed budget N for data reuse)")
    vectors = np.load(str(vec_path))   # full load — we need float32 to build index
    queries = np.load(str(qry_path))
    return vectors, queries


# ---------------------------------------------------------------------------
# Pressure profiles
# ---------------------------------------------------------------------------

def make_ramp_profile(start_pct, end_pct, ramp_seconds):
    def _p(elapsed):
        return start_pct + (end_pct - start_pct) * min(elapsed / ramp_seconds, 1.0)
    return _p


def make_spike_profile(baseline_pct, peak_pct, rise_s, hold_s, fall_s, idle_s):
    period = rise_s + hold_s + fall_s + idle_s
    def _p(elapsed):
        phase = elapsed % period
        if phase < rise_s:
            return baseline_pct + (peak_pct - baseline_pct) * (phase / rise_s)
        phase -= rise_s
        if phase < hold_s:
            return peak_pct
        phase -= hold_s
        if phase < fall_s:
            return peak_pct + (baseline_pct - peak_pct) * (phase / fall_s)
        return baseline_pct
    return _p


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def maybe_write_csv(path: Optional[str], rows: List[dict]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_index", "elapsed_sec",
        "target_pressure_pct", "actual_pressure_pct",
        "latency_ms",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ── Index / data ──────────────────────────────────────────────────────
    g = parser.add_argument_group("index / data")
    g.add_argument(
        "--index-type", choices=["int8", "float32"], default="float32",
        help="Index type: 'float32' (IndexHNSWFlat) or 'int8' (IndexHNSWSQ QT_8bit) (default: float32)",
    )
    g.add_argument(
        "--total-index-gb", type=float, default=1.0,
        help="Memory budget for the index in GiB (default: 1.0)",
    )
    g.add_argument("--dim", type=int, default=128, help="Embedding dimension (default: 128)")
    g.add_argument("--m", type=int, default=16, help="HNSW M parameter (default: 16)")
    g.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction (default: 200)")
    g.add_argument("--ef-search", type=int, default=64, help="HNSW ef_search (default: 64)")
    g.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    g.add_argument(
        "--index-cache-dir", default=None,
        help="Directory to cache the built index (default: index_cache_float32 or index_cache_int8)",
    )
    g.add_argument(
        "--shared-data-dir", default=None,
        help="Optional path to a benchmark_faiss_twophase.py cache directory. "
             "Reuses its vectors.npy and queries.npy so both benchmarks run on "
             "the exact same dataset. Only the FAISS index is rebuilt.",
    )
    g.add_argument("--rebuild", action="store_true", help="Force index rebuild even if cache exists")

    # ── Query ─────────────────────────────────────────────────────────────
    g2 = parser.add_argument_group("query")
    g2.add_argument("--top-k", type=int, default=10, help="Number of results per query (default: 10)")
    g2.add_argument("--duration-seconds", type=float, default=120.0, help="Benchmark runtime in seconds (default: 120)")
    g2.add_argument("--query-pool-size", type=int, default=1000, help="Number of pre-generated queries (default: 1000)")
    g2.add_argument("--query-interval-ms", type=float, default=0.0, help="Optional sleep between queries in ms (default: 0)")

    # ── Memory pressure ──────────────────────────────────────────────────
    g3 = parser.add_argument_group("memory pressure")
    g3.add_argument("--pressure-profile", choices=["ramp", "spike"], default="ramp")
    g3.add_argument("--pressure-start-pct", type=float, default=0.0)
    g3.add_argument("--pressure-end-pct", type=float, default=75.0)
    g3.add_argument("--ramp-seconds", type=float, default=120.0)
    g3.add_argument("--spike-baseline-pct", type=float, default=0.0)
    g3.add_argument("--spike-peak-pct", type=float, default=70.0)
    g3.add_argument("--spike-rise-seconds", type=float, default=5.0)
    g3.add_argument("--spike-hold-seconds", type=float, default=10.0)
    g3.add_argument("--spike-fall-seconds", type=float, default=5.0)
    g3.add_argument("--spike-idle-seconds", type=float, default=20.0)
    g3.add_argument("--chunk-mb", type=int, default=64)
    g3.add_argument("--allocator-pause-ms", type=float, default=25.0)
    g3.add_argument("--read-segment-pct", type=float, default=0.0)
    g3.add_argument("--read-interval-ms", type=float, default=100.0)

    # ── Output ────────────────────────────────────────────────────────────
    g4 = parser.add_argument_group("output")
    g4.add_argument("--csv-out", default=None,
                    help="CSV output path (default: outputs/baseline_{index_type}.csv)")
    g4.add_argument("--plot-window", type=int, default=200)

    args = parser.parse_args()

    # Defaults that depend on --index-type
    if args.index_cache_dir is None:
        args.index_cache_dir = f"index_cache_{args.index_type}"
    if args.csv_out is None:
        args.csv_out = f"outputs/baseline_{args.index_type}.csv"

    # ── Setup ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # If sharing data with the two-phase cache, derive N from that dataset so
    # the cache key is consistent across both scripts.
    n_vectors = compute_num_vectors(args.total_index_gb, args.dim, args.m, args.index_type)
    if args.shared_data_dir:
        _meta_path = Path(args.shared_data_dir) / "meta.json"
        try:
            with _meta_path.open() as _f:
                _shared_meta = json.load(_f)
            if _shared_meta.get("dim") == args.dim and _shared_meta.get("seed") == args.seed:
                n_vectors = _shared_meta["n_vectors"]
        except Exception:
            pass

    print_memory_budget(n_vectors, args.dim, args.m, args.index_type)

    mem_total_bytes = get_mem_total_bytes()
    print(f"System/cgroup total RAM : {mem_total_bytes / (1024**3):.2f} GiB")

    # ── Load or build index ───────────────────────────────────────────────
    cache_dir = Path(args.index_cache_dir)
    loaded = None if args.rebuild else try_load_cache(
        cache_dir, n_vectors, args.dim, args.m, args.ef_construction,
        args.seed, args.ef_search, args.index_type,
    )

    if loaded is not None:
        index, vectors, query_pool = loaded
    else:
        # Try to reuse data (vectors + queries) from a two-phase cache
        shared = None
        if args.shared_data_dir:
            shared = try_load_shared_data(
                Path(args.shared_data_dir), args.dim, args.seed
            )

        if shared is not None:
            vectors, query_pool = shared
        else:
            print(f"Generating {n_vectors:,} × dim-{args.dim} float32 vectors…")
            vectors = generate_vectors(n_vectors, args.dim, rng)
            print(f"Generating {args.query_pool_size} query vectors…")
            query_pool = generate_vectors(args.query_pool_size, args.dim, rng)

        index = build_index(vectors, args.index_type, args.m, args.ef_construction, args.ef_search)
        save_cache(
            cache_dir, index, vectors, query_pool,
            n_vectors, args.dim, args.m, args.ef_construction, args.seed, args.index_type,
        )

    # ── Pressure profile ──────────────────────────────────────────────────
    if args.pressure_profile == "ramp":
        profile_fn = make_ramp_profile(args.pressure_start_pct, args.pressure_end_pct, args.ramp_seconds)
    else:
        profile_fn = make_spike_profile(
            args.spike_baseline_pct, args.spike_peak_pct,
            args.spike_rise_seconds, args.spike_hold_seconds,
            args.spike_fall_seconds, args.spike_idle_seconds,
        )

    # ── Allocator process ─────────────────────────────────────────────────
    target_bytes = mp.Value("Q", 0)
    actual_bytes = mp.Value("Q", 0)
    stop_event = mp.Event()
    pressure_proc = mp.Process(
        target=memory_pressure_worker,
        args=(
            target_bytes, actual_bytes, stop_event,
            args.chunk_mb * 1024 * 1024,
            args.allocator_pause_ms / 1000.0,
            args.read_segment_pct,
            args.read_interval_ms / 1000.0,
        ),
        daemon=True,
    )

    # ── Query loop ─────────────────────────────────────────────────────────
    rows: List[dict] = []
    latencies: List[float] = []

    pressure_proc.start()
    start = time.perf_counter()
    next_report = 50
    query_index = 0
    q_buf = np.empty((1, args.dim), dtype=np.float32)

    print(f"\nStarting {args.index_type} baseline  [{args.duration_seconds:.0f}s, {args.pressure_profile} profile]")
    print()

    try:
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
            actual_pct = (actual_b / mem_total_bytes) * 100.0 if mem_total_bytes else 0.0

            q_buf[0] = query_pool[query_index % args.query_pool_size]

            t0 = time.perf_counter()
            index.search(q_buf, args.top_k)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            latencies.append(latency_ms)
            query_index += 1

            rows.append({
                "query_index":        query_index,
                "elapsed_sec":        round(elapsed, 4),
                "target_pressure_pct": round(target_pct, 3),
                "actual_pressure_pct": round(actual_pct, 3),
                "latency_ms":         round(latency_ms, 4),
            })

            if query_index >= next_report:
                recent = latencies[max(0, len(latencies) - 50):]
                s = sorted(recent)
                print(
                    f"q={query_index:>7,} | t={elapsed:6.1f}s | "
                    f"target={target_pct:5.1f}% | actual={actual_pct:5.1f}% | "
                    f"p50={percentile(s, 50):.3f}ms p99={percentile(s, 99):.3f}ms"
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
        raise RuntimeError("No queries executed")

    sorted_lat = sorted(latencies)
    avg_lat = statistics.mean(sorted_lat)
    p50 = percentile(sorted_lat, 50)
    p99 = percentile(sorted_lat, 99)

    print(f"\n{'═'*52}")
    print(f"  FAISS {args.index_type} Baseline Results")
    print(f"{'═'*52}")
    print(f"  Index type        : {args.index_type}")
    print(f"  Queries executed  : {len(sorted_lat):>10,}")
    print(f"  Latency avg       : {avg_lat:>10.3f} ms")
    print(f"  Latency P50       : {p50:>10.3f} ms")
    print(f"  Latency P99       : {p99:>10.3f} ms")
    print(f"  Latency min/max   : {sorted_lat[0]:.3f} / {sorted_lat[-1]:.3f} ms")
    print(f"{'═'*52}")

    maybe_write_csv(args.csv_out, rows)
    if args.csv_out:
        print(f"\nWrote CSV: {args.csv_out}")
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
                ],
                check=True,
            )
            print(f"Plot saved: {png_path}")
        except Exception as exc:
            print(f"Warning: plot generation failed: {exc}")


if __name__ == "__main__":
    main()
