"""benchmark_faiss_twophase.py

Two-pass FAISS benchmark for evaluating the memory-pressure trade-off between
search quality and latency.

Architecture
------------
  Pass 1  (always):   INT8 HNSW coarse search → candidate set of size (top_k * candidate_mult)
  Pass 2  (adaptive): Fetch the corresponding float32 vectors and re-rank by exact L2 distance.
                      Skipped when actual memory pressure >= --rerank-threshold-pct.

Memory layout
-------------
  float32 store   : N × dim × 4 bytes   (mmap'd — only touched pages loaded)
  int8 SQ vectors : N × dim × 1 byte
  HNSW graph      : N × 2 × M × 4 bytes  (per-node neighbor lists, approximate)
  ─────────────────────────────────────────
  total           ≈ N × (5×dim + 8×M) bytes

Output CSV is compatible with plot_continuous_results.py (required columns:
elapsed_sec, latency_ms, target_pressure_pct, actual_pressure_pct).

Usage
-----
Build the index first with build_faiss_indexes.py, then run this benchmark:

  # Build (once — chunked, only one chunk of RAM at a time)
  python build_faiss_indexes.py --type twophase --total-index-gb 0.5 --dim 128

  # Run with ramp pressure
  python benchmark_faiss_twophase.py --dim 128 \\
      --pressure-end-pct 75 --duration-seconds 120

  # Run with spike pressure
  python benchmark_faiss_twophase.py --dim 1536 \\
      --pressure-profile spike --spike-peak-pct 80 --duration-seconds 300 \\
      --csv-out outputs/twophase_spike.csv
"""
import argparse
import csv
import json
import multiprocessing as mp
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Utilities (shared with the continuous benchmarks)
# ---------------------------------------------------------------------------

def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("No values to compute percentile")
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
    """Total memory available, respecting cgroup limits when present."""
    try:
        with open("/sys/fs/cgroup/chromabench/memory.max", encoding="utf-8") as f:
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
    target_bytes: object,
    actual_bytes: object,
    stop_event: object,
    chunk_bytes: int,
    pause_seconds: float,
    read_segment_pct: float = 0.0,
    read_interval_seconds: float = 0.1,
) -> None:
    """Allocator process — identical to the continuous benchmark workers."""
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
                _ = current_chunk[read_chunk_offset : read_chunk_offset + to_read]
                read_chunk_offset += to_read
                bytes_remaining -= to_read
                if read_chunk_offset >= len(current_chunk):
                    read_chunk_idx = (read_chunk_idx + 1) % len(chunks)
                    read_chunk_offset = 0

        time.sleep(pause_seconds)


# ---------------------------------------------------------------------------
# Memory-budget sizing
# ---------------------------------------------------------------------------

def print_memory_budget(n: int, dim: int, m: int) -> None:
    float32_gb = n * dim * 4 / (1024 ** 3)
    int8_gb    = n * dim * 1 / (1024 ** 3)
    graph_gb   = n * 2 * m * 4 / (1024 ** 3)
    total_gb   = float32_gb + int8_gb + graph_gb
    print(f"\n{'─'*52}")
    print(f"  Index memory budget")
    print(f"{'─'*52}")
    print(f"  Vectors (N)     : {n:>12,}")
    print(f"  Dimension       : {dim:>12,}")
    print(f"  HNSW M          : {m:>12,}")
    print(f"  float32 store   : {float32_gb:>11.3f} GiB   ({n * dim * 4 / 1e9:.2f} GB)")
    print(f"  int8 SQ vectors : {int8_gb:>11.3f} GiB   ({n * dim * 1 / 1e9:.2f} GB)")
    print(f"  HNSW graph      : {graph_gb:>11.3f} GiB   ({n * 2 * m * 4 / 1e9:.2f} GB)")
    print(f"  ─────────────────────────────────────")
    print(f"  Total (approx.) : {total_gb:>11.3f} GiB")
    print(f"{'─'*52}\n")


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def _cache_meta(n: int, dim: int, m: int, ef_construction: int, seed: int) -> dict:
    return {"n_vectors": n, "dim": dim, "m": m, "ef_construction": ef_construction, "seed": seed}


def _cache_paths(cache_dir: "Path"):
    return (
        cache_dir / "index.faiss",
        cache_dir / "vectors.npy",
        cache_dir / "queries.npy",
        cache_dir / "meta.json",
    )


def try_load_cache(
    cache_dir: "Path", dim: int, m: int, ef_construction: int, seed: int, ef_search: int
) -> Optional[tuple]:
    """Return (int8_index, float32_store, query_pool, n_vectors) from cache, or None if missing/invalid.

    float32_store is memory-mapped (read-only), so only touched pages are loaded
    into RAM — useful for studying OS page-eviction behaviour under pressure.
    Build the cache first with: python build_faiss_indexes.py --type twophase ...
    """
    idx_path, vec_path, qry_path, meta_path = _cache_paths(cache_dir)
    if not all(p.exists() for p in (idx_path, vec_path, qry_path, meta_path)):
        return None
    try:
        with meta_path.open() as f:
            saved = json.load(f)
    except Exception:
        return None
    if (saved.get("dim") != dim or saved.get("m") != m
            or saved.get("ef_construction") != ef_construction
            or saved.get("seed") != seed):
        print(f"Cache found in '{cache_dir}' but parameters differ.")
        print(f"  Cached:    dim={saved.get('dim')}, m={saved.get('m')}, "
              f"ef_construction={saved.get('ef_construction')}, seed={saved.get('seed')}")
        print(f"  Requested: dim={dim}, m={m}, ef_construction={ef_construction}, seed={seed}")
        return None
    n_vectors = saved["n_vectors"]
    print(f"Loading cached index from: {cache_dir}")
    int8_index = faiss.read_index(str(idx_path))
    int8_index.hnsw.efSearch = ef_search
    float32_store = np.load(str(vec_path), mmap_mode="r")
    query_pool = np.load(str(qry_path))
    print(f"  Loaded {int8_index.ntotal:,} vectors  (float32 store memory-mapped)")
    return int8_index, float32_store, query_pool, n_vectors


# ---------------------------------------------------------------------------
# Two-pass query
# ---------------------------------------------------------------------------

def two_pass_query(
    int8_index: faiss.IndexHNSWSQ,
    float32_store: np.ndarray,
    query: np.ndarray,
    top_k: int,
    candidate_mult: int,
    rerank: bool,
) -> Tuple[np.ndarray, float, float, float]:
    """Execute one query and return (result_ids, total_ms, pass1_ms, pass2_ms).

    Parameters
    ----------
    int8_index      : Trained IndexHNSWSQ
    float32_store   : (N, dim) float32 array — original (unreduced) vectors
    query           : (dim,) float32 query vector
    top_k           : Final number of results to return
    candidate_mult  : Pass-1 oversampling: fetch top_k * candidate_mult candidates
    rerank          : True → run pass 2; False → return pass-1 results directly
    """
    q = query.reshape(1, -1).astype(np.float32)

    # ── Pass 1: coarse int8 search ──────────────────────────────────────────
    n_candidates = top_k * candidate_mult if rerank else top_k
    t0 = time.perf_counter()
    _, ids = int8_index.search(q, n_candidates)
    pass1_ms = (time.perf_counter() - t0) * 1000.0

    ids = ids[0]
    valid_ids = ids[ids >= 0]  # FAISS returns -1 for unfilled slots

    if not rerank or len(valid_ids) == 0:
        total_ms = pass1_ms
        return valid_ids[:top_k], total_ms, pass1_ms, 0.0

    # ── Pass 2: exact float32 re-rank ───────────────────────────────────────
    t1 = time.perf_counter()
    candidate_vecs = float32_store[valid_ids]            # (C, dim)
    diffs = candidate_vecs - query[np.newaxis, :]        # (C, dim)
    dists = np.einsum("ij,ij->i", diffs, diffs)          # (C,) L2²
    order = np.argsort(dists)
    result_ids = valid_ids[order[:top_k]]
    pass2_ms = (time.perf_counter() - t1) * 1000.0

    total_ms = pass1_ms + pass2_ms
    return result_ids, total_ms, pass1_ms, pass2_ms


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def maybe_write_csv(path: Optional[str], rows: List[dict]) -> None:
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
        "pass1_ms",
        "pass2_ms",
        "reranked",
        "rerank_skipped",
        "n_candidates",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Pressure profiles (identical to continuous benchmarks)
# ---------------------------------------------------------------------------

def make_ramp_profile(start_pct: float, end_pct: float, ramp_seconds: float):
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
    g.add_argument("--dim", type=int, default=None,
                   help="Embedding dimension (resolved from cache metadata if omitted)")
    g.add_argument("--m", type=int, default=None,
                   help="HNSW M parameter (resolved from cache metadata if omitted)")
    g.add_argument("--ef-construction", type=int, default=None,
                   help="HNSW ef_construction (resolved from cache metadata if omitted)")
    g.add_argument("--ef-search", type=int, default=64,
                   help="HNSW ef_search — query-time beam width (default: 64)")
    g.add_argument("--seed", type=int, default=None,
                   help="Random seed (resolved from cache metadata if omitted)")
    g.add_argument(
        "--index-cache-dir", default="index_cache",
        help="Path to the pre-built index cache created by build_faiss_indexes.py (default: index_cache)",
    )

    # ── Two-pass behaviour ───────────────────────────────────────────────
    g2 = parser.add_argument_group("two-pass behaviour")
    g2.add_argument("--top-k", type=int, default=10,
                    help="Final number of results returned per query (default: 10)")
    g2.add_argument(
        "--candidate-mult", type=int, default=5,
        help="Pass-1 oversampling factor: fetch top_k × candidate_mult candidates "
             "before exact re-rank (default: 5)",
    )
    g2.add_argument(
        "--rerank-threshold-pct", type=float, default=60.0,
        help="Skip pass-2 re-rank when actual memory pressure exceeds this %% of "
             "total RAM (default: 60.0). Set to 0 to always skip; 100 to always rerank.",
    )
    # ── Benchmark timing ─────────────────────────────────────────────────
    g3 = parser.add_argument_group("benchmark timing")
    g3.add_argument("--duration-seconds", type=float, default=120.0,
                    help="Total benchmark runtime in seconds (default: 120)")
    g3.add_argument(
        "--query-source", choices=["cache", "random"], default="cache",
        help="'cache' uses queries.npy from the index cache; 'random' generates fresh unit-sphere vectors (default: cache)",
    )
    g3.add_argument("--query-pool-size", type=int, default=1000,
                    help="Number of pre-generated query vectors (default: 1000)")
    g3.add_argument("--query-interval-ms", type=float, default=0.0,
                    help="Optional sleep between queries in ms (default: 0)")

    # ── Memory pressure ──────────────────────────────────────────────────
    g4 = parser.add_argument_group("memory pressure")
    g4.add_argument(
        "--pressure-profile", choices=["ramp", "spike"], default="ramp",
        help="Pressure shape: 'ramp' (linear, default) or 'spike' (periodic pulses)",
    )
    g4.add_argument("--pressure-start-pct", type=float, default=0.0,
                    help="[ramp] Starting allocator pressure %% of total RAM (default: 0)")
    g4.add_argument("--pressure-end-pct", type=float, default=75.0,
                    help="[ramp] Ending allocator pressure %% of total RAM (default: 75)")
    g4.add_argument("--ramp-seconds", type=float, default=120.0,
                    help="[ramp] How long to complete the ramp (default: 120 s)")
    g4.add_argument("--spike-baseline-pct", type=float, default=0.0,
                    help="[spike] Pressure %% at idle (default: 0)")
    g4.add_argument("--spike-peak-pct", type=float, default=70.0,
                    help="[spike] Peak pressure %% (default: 70)")
    g4.add_argument("--spike-rise-seconds", type=float, default=5.0,
                    help="[spike] Seconds to ramp baseline→peak (default: 5)")
    g4.add_argument("--spike-hold-seconds", type=float, default=10.0,
                    help="[spike] Seconds to hold peak (default: 10)")
    g4.add_argument("--spike-fall-seconds", type=float, default=5.0,
                    help="[spike] Seconds to fall peak→baseline (default: 5)")
    g4.add_argument("--spike-idle-seconds", type=float, default=20.0,
                    help="[spike] Seconds to idle at baseline before repeating (default: 20)")
    g4.add_argument("--chunk-mb", type=int, default=64,
                    help="Allocator chunk size in MiB (default: 64)")
    g4.add_argument("--allocator-pause-ms", type=float, default=25.0,
                    help="Allocator sleep between allocation checks in ms (default: 25)")
    g4.add_argument(
        "--read-segment-pct", type=float, default=0.0,
        help="Rolling read-through: scan this %% of allocated memory per cycle "
             "to keep pages active and compete with the query workload (default: 0 = off)",
    )
    g4.add_argument("--read-interval-ms", type=float, default=100.0,
                    help="Minimum time between read sweeps in ms (default: 100)")

    # ── Output ────────────────────────────────────────────────────────────
    g5 = parser.add_argument_group("output")
    g5.add_argument(
        "--csv-out", default="outputs/twophase_results.csv",
        help="CSV file for per-query results (default: outputs/twophase_results.csv)",
    )
    g5.add_argument("--plot-window", type=int, default=200,
                    help="Rolling window size for the auto-generated plot (default: 200)")

    args = parser.parse_args()

    # ── Resolve missing args from cache metadata ────────────────────────────
    if any(v is None for v in [args.dim, args.m, args.ef_construction, args.seed]):
        _meta_path = Path(args.index_cache_dir) / "meta.json"
        try:
            with _meta_path.open() as _f:
                _meta = json.load(_f)
            for _key in ("dim", "m", "ef_construction", "seed"):
                if _key not in _meta:
                    raise KeyError(_key)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as _exc:
            raise RuntimeError(
                f"Could not read '{_meta_path}' to resolve missing arguments.\n"
                f"Provide --dim, --m, --ef-construction, and --seed explicitly, or "
                f"build the cache first with:\n"
                f"  python build_faiss_indexes.py --type twophase ..."
            ) from _exc
        if args.dim            is None: args.dim            = _meta["dim"]
        if args.m              is None: args.m              = _meta["m"]
        if args.ef_construction is None: args.ef_construction = _meta["ef_construction"]
        if args.seed           is None: args.seed           = _meta["seed"]

    # ── Validation ────────────────────────────────────────────────────────
    if args.dim < 2:
        raise ValueError("--dim must be >= 2")
    if args.m < 2:
        raise ValueError("--m must be >= 2")
    if args.ef_construction < args.m:
        raise ValueError("--ef-construction must be >= --m")
    if args.ef_search < 1:
        raise ValueError("--ef-search must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.candidate_mult < 1:
        raise ValueError("--candidate-mult must be >= 1")
    if not (0.0 <= args.rerank_threshold_pct <= 100.0):
        raise ValueError("--rerank-threshold-pct must be in [0, 100]")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0")
    if args.query_pool_size < 1:
        raise ValueError("--query-pool-size must be >= 1")
    if args.query_interval_ms < 0:
        raise ValueError("--query-interval-ms must be >= 0")
    if args.pressure_start_pct < 0 or args.pressure_start_pct > 95:
        raise ValueError("--pressure-start-pct must be in [0, 95]")
    if args.pressure_end_pct < 0 or args.pressure_end_pct > 95:
        raise ValueError("--pressure-end-pct must be in [0, 95]")
    if args.ramp_seconds <= 0:
        raise ValueError("--ramp-seconds must be > 0")
    if args.chunk_mb < 1:
        raise ValueError("--chunk-mb must be >= 1")
    if args.read_segment_pct < 0 or args.read_segment_pct > 100:
        raise ValueError("--read-segment-pct must be in [0, 100]")
    if args.read_interval_ms <= 0:
        raise ValueError("--read-interval-ms must be > 0")
    if args.pressure_profile == "spike":
        if not (0 <= args.spike_baseline_pct <= 95):
            raise ValueError("--spike-baseline-pct must be in [0, 95]")
        if not (args.spike_baseline_pct < args.spike_peak_pct <= 95):
            raise ValueError("--spike-peak-pct must be > --spike-baseline-pct and <= 95")
        for flag, val in [
            ("--spike-rise-seconds", args.spike_rise_seconds),
            ("--spike-hold-seconds", args.spike_hold_seconds),
            ("--spike-fall-seconds", args.spike_fall_seconds),
        ]:
            if val <= 0:
                raise ValueError(f"{flag} must be > 0")
        if args.spike_idle_seconds < 0:
            raise ValueError("--spike-idle-seconds must be >= 0")

    # ── Setup ─────────────────────────────────────────────────────────────
    mem_total_bytes = get_mem_total_bytes()
    print(f"System/cgroup total RAM : {mem_total_bytes / (1024**3):.2f} GiB")

    # ── Load from cache ────────────────────────────────────────────────────
    cache_dir = Path(args.index_cache_dir)
    loaded = try_load_cache(
        cache_dir, args.dim, args.m, args.ef_construction, args.seed, args.ef_search
    )
    if loaded is None:
        raise RuntimeError(
            f"No valid index cache found in '{cache_dir}'.\n"
            f"Build it first with:\n"
            f"  python build_faiss_indexes.py --type twophase "
            f"--total-index-gb <GB> --dim {args.dim} --m {args.m} "
            f"--ef-construction {args.ef_construction} --seed {args.seed}"
        )
    int8_index, float32_store, query_pool, n_vectors = loaded
    print_memory_budget(n_vectors, args.dim, args.m)

    # ── Query pool ────────────────────────────────────────────────────────
    if args.query_source == "random":
        rng = np.random.default_rng(args.seed if args.seed is not None else 42)
        vecs = rng.standard_normal((args.query_pool_size, args.dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        query_pool = (vecs / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        print(f"  Generated {args.query_pool_size:,} random query vectors (unit sphere)")
    else:
        if len(query_pool) < args.query_pool_size:
            print(f"  Note: cache contains {len(query_pool):,} queries; "
                  f"--query-pool-size capped to {len(query_pool):,}")
            args.query_pool_size = len(query_pool)
    query_pool_len = len(query_pool)

    # ── Pressure profile ──────────────────────────────────────────────────
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

    # ── Allocator process ─────────────────────────────────────────────────
    target_bytes = mp.Value("Q", 0)
    actual_bytes = mp.Value("Q", 0)
    stop_event = mp.Event()

    pressure_proc = mp.Process(
        target=memory_pressure_worker,
        args=(
            target_bytes,
            actual_bytes,
            stop_event,
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
    rerank_count = 0
    pass1_only_count = 0

    pressure_proc.start()
    start = time.perf_counter()
    next_report = 50
    query_index = 0

    print(f"\nStarting benchmark  [{args.duration_seconds:.0f}s, {args.pressure_profile} profile]")
    print(f"  Rerank skipped when actual pressure >= {args.rerank_threshold_pct:.1f}%")
    print(f"  Candidate multiplier: {args.candidate_mult}× ({args.top_k}→{args.top_k * args.candidate_mult} candidates)")
    print()

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= args.duration_seconds:
                break

            # Update allocator target
            target_pct = profile_fn(elapsed)
            target_b = int((target_pct / 100.0) * mem_total_bytes)
            with target_bytes.get_lock():
                target_bytes.value = target_b
            with actual_bytes.get_lock():
                actual_b = int(actual_bytes.value)

            actual_pct = (actual_b / mem_total_bytes) * 100.0 if mem_total_bytes else 0.0

            # Decide whether to rerank
            do_rerank = actual_pct < args.rerank_threshold_pct

            query_vec = query_pool[query_index % query_pool_len]

            result_ids, total_ms, pass1_ms, pass2_ms = two_pass_query(
                int8_index, float32_store, query_vec,
                args.top_k, args.candidate_mult, do_rerank,
            )

            latencies.append(total_ms)
            if do_rerank:
                rerank_count += 1
            else:
                pass1_only_count += 1
            query_index += 1

            rows.append({
                "query_index": query_index,
                "elapsed_sec": round(elapsed, 4),
                "target_pressure_pct": round(target_pct, 3),
                "actual_pressure_pct": round(actual_pct, 3),
                "latency_ms": round(total_ms, 4),
                "pass1_ms": round(pass1_ms, 4),
                "pass2_ms": round(pass2_ms, 4),
                "reranked": "yes" if do_rerank else "no",
                "rerank_skipped": "yes" if not do_rerank else "no",
                "n_candidates": len(result_ids),
            })

            if query_index >= next_report:
                recent = latencies[max(0, len(latencies) - 50):]
                s = sorted(recent)
                p50 = percentile(s, 50)
                p99 = percentile(s, 99)
                print(
                    f"q={query_index:>7,} | t={elapsed:6.1f}s | "
                    f"target={target_pct:5.1f}% | actual={actual_pct:5.1f}% | "
                    f"rerank={'Y' if do_rerank else 'N'} | "
                    f"p50={p50:.3f}ms p99={p99:.3f}ms"
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
    avg_lat = statistics.mean(sorted_lat)
    p50 = percentile(sorted_lat, 50)
    p99 = percentile(sorted_lat, 99)

    print(f"\n{'═'*52}")
    print("  Two-Phase FAISS Benchmark Results")
    print(f"{'═'*52}")
    print(f"  Queries executed  : {len(sorted_lat):>10,}")
    print(f"  Two-pass (rerank) : {rerank_count:>10,}  ({100*rerank_count/len(sorted_lat):.1f}%)")
    print(f"  Pass-1 only       : {pass1_only_count:>10,}  ({100*pass1_only_count/len(sorted_lat):.1f}%)")
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
                    "--latency-y-max", "15",
                ],
                check=True,
            )
            print(f"Plot saved: {png_path}")
        except Exception as exc:
            print(f"Warning: plot generation failed: {exc}")


if __name__ == "__main__":
    main()
