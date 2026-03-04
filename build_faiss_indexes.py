"""build_faiss_indexes.py

Pre-build and cache FAISS indexes for use with benchmark_faiss_twophase.py
and benchmark_faiss_baseline.py.  Designed for large indexes (tens of GBs)
where generating all vectors in RAM at once is not feasible.

Key differences from the in-benchmark build path
-------------------------------------------------
* Vector generation is chunked: only --build-chunk-size vectors are held in
  RAM at one time.  The full float32 array is streamed to disk via a
  memory-mapped .npy file (np.lib.format.open_memmap), so peak RAM usage is
  bounded by the chunk size, not by N.

* FAISS index.add() is also chunked (same chunk size), so the index grows
  incrementally.

* The SQ training sample is drawn from the first chunk (capped at 50 K).

* The output cache directories and meta.json formats are identical to those
  expected by benchmark_faiss_twophase.py and benchmark_faiss_baseline.py,
  so the benchmark scripts will load the pre-built cache without rebuilding.

Index types
-----------
  twophase   IndexHNSWSQ (QT_8bit) + vectors.npy + queries.npy
             → cache dir: --twophase-cache-dir  (default: index_cache)
             → compatible with: benchmark_faiss_twophase.py

  float32    IndexHNSWFlat (float32) + vectors.npy + queries.npy
             → cache dir: --float32-cache-dir   (default: index_cache_float32)
             → compatible with: benchmark_faiss_baseline.py --index-type float32

  int8       IndexHNSWSQ (QT_8bit) + vectors.npy + queries.npy
             → cache dir: --int8-cache-dir      (default: index_cache_int8)
             → compatible with: benchmark_faiss_baseline.py --index-type int8

  all        Build all three.  vectors.npy and queries.npy are generated once
             and symlinked (or hard-copied) into the float32 and int8 dirs so
             --shared-data-dir still works correctly.

Memory budget (informational — N is set explicitly via --n-vectors or via
--total-index-gb using the same formula as the benchmark scripts)
---------------------------------------------------------------------------
  two-phase  : N × (5×dim + 8×M) bytes  (float32 store + int8 SQ + graph)
  float32    : N × (4×dim + 8×M) bytes  (float32 inside index + graph)
  int8       : N × (  dim + 8×M) bytes  (int8 SQ + graph)

Usage examples
--------------
  # 80 GB two-phase index, dim=1536, chunk 2 GB at a time
  python build_faiss_indexes.py --type twophase \\
      --total-index-gb 80 --dim 1536 --build-chunk-size 500000

  # All three indexes on the same dataset (vectors generated once)
  python build_faiss_indexes.py --type all \\
      --total-index-gb 80 --dim 1536 --build-chunk-size 500000

  # Explicit N instead of memory budget
  python build_faiss_indexes.py --type float32 --n-vectors 10000000 --dim 128
"""
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Memory-budget sizing  (mirrors both benchmark scripts)
# ---------------------------------------------------------------------------

def compute_n_twophase(total_gb: float, dim: int, m: int) -> int:
    """N for two-phase: float32 store + int8 SQ + HNSW graph."""
    return max(1000, int(total_gb * (1024 ** 3) / (5 * dim + 8 * m)))


def compute_n_float32(total_gb: float, dim: int, m: int) -> int:
    """N for float32 HNSW: vectors inside index + graph."""
    return max(1000, int(total_gb * (1024 ** 3) / (4 * dim + 8 * m)))


def compute_n_int8(total_gb: float, dim: int, m: int) -> int:
    """N for int8-only HNSW: int8 SQ + graph."""
    return max(1000, int(total_gb * (1024 ** 3) / (dim + 8 * m)))


def print_budget(n: int, dim: int, m: int, index_type: str) -> None:
    if index_type == "twophase":
        components = [
            ("float32 store  ", n * dim * 4),
            ("int8 SQ vectors", n * dim * 1),
            ("HNSW graph     ", n * 2 * m * 4),
        ]
        label = "two-phase"
    elif index_type == "float32":
        components = [
            ("float32 vectors", n * dim * 4),
            ("HNSW graph     ", n * 2 * m * 4),
        ]
        label = "float32 HNSW"
    else:
        components = [
            ("int8 SQ vectors", n * dim * 1),
            ("HNSW graph     ", n * 2 * m * 4),
        ]
        label = "int8 HNSW"

    total = sum(b for _, b in components)
    print(f"\n{'─'*54}")
    print(f"  Index memory budget  [{label}]")
    print(f"{'─'*54}")
    print(f"  Vectors (N) : {n:>14,}")
    print(f"  Dimension   : {dim:>14,}")
    print(f"  HNSW M      : {m:>14,}")
    for name, b in components:
        print(f"  {name} : {b / (1024**3):>10.3f} GiB")
    print(f"  {'─'*38}")
    print(f"  Total       : {total / (1024**3):>10.3f} GiB")
    print(f"{'─'*54}\n")


# ---------------------------------------------------------------------------
# Chunked vector generation
# ---------------------------------------------------------------------------

def generate_and_save_vectors_chunked(
    n: int,
    dim: int,
    rng: np.random.Generator,
    out_path: Path,
    chunk_size: int,
) -> None:
    """Generate N × dim float32 L2-normalised vectors in chunks.

    Uses np.lib.format.open_memmap to pre-allocate the .npy file on disk
    and fill it chunk-by-chunk.  Peak RAM = one chunk, not N × dim × 4 bytes.
    """
    print(f"  Pre-allocating {out_path}  ({n * dim * 4 / (1024**3):.2f} GiB on disk)…")
    fp = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32, shape=(n, dim))

    written = 0
    t0 = time.monotonic()
    while written < n:
        size = min(chunk_size, n - written)
        chunk = rng.standard_normal((size, dim)).astype(np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        chunk /= norms
        fp[written : written + size] = chunk
        written += size
        elapsed = time.monotonic() - t0
        pct = 100.0 * written / n
        rate = written / elapsed if elapsed > 0 else 0
        print(
            f"    generated {written:>12,} / {n:,}  ({pct:5.1f}%)  "
            f"{rate / 1e6:.2f}M vec/s",
            end="\r",
        )
    print()  # newline after \r progress
    fp.flush()
    del fp
    print(f"  Saved {out_path}  ({n * dim * 4 / (1024**3):.2f} GiB)")


def generate_queries(
    n: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n query vectors fully in RAM (small — typically 1 K vectors)."""
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Index construction (chunked add)
# ---------------------------------------------------------------------------

def build_hnsw_sq(
    vec_path: Path,
    n: int,
    dim: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    chunk_size: int,
) -> faiss.IndexHNSWSQ:
    """Build IndexHNSWSQ from an on-disk float32 .npy file, adding in chunks."""
    print(f"  Building IndexHNSWSQ (QT_8bit, M={m}, ef_construction={ef_construction})…")
    index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    # Load vectors via mmap for chunked access without pre-loading everything
    store = np.lib.format.open_memmap(str(vec_path), mode="r")

    # Train on first chunk (capped at 50 K)
    train_n = min(n, 50_000, chunk_size)
    print(f"  Training scalar quantizer on {train_n:,} vectors…")
    index.train(store[:train_n])

    # Add in chunks
    print(f"  Adding {n:,} vectors in chunks of {chunk_size:,}…")
    added = 0
    t0 = time.monotonic()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        index.add(np.array(store[start:end]))  # np.array forces a copy out of mmap
        added = end
        elapsed = time.monotonic() - t0
        pct = 100.0 * added / n
        rate = added / elapsed if elapsed > 0 else 0
        print(
            f"    added {added:>12,} / {n:,}  ({pct:5.1f}%)  "
            f"{rate / 1e6:.2f}M vec/s",
            end="\r",
        )
    print()

    del store
    print(f"  Index built: {index.ntotal:,} vectors")
    return index


def build_hnsw_flat(
    vec_path: Path,
    n: int,
    dim: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    chunk_size: int,
) -> faiss.IndexHNSWFlat:
    """Build IndexHNSWFlat from an on-disk float32 .npy file, adding in chunks."""
    print(f"  Building IndexHNSWFlat (float32, M={m}, ef_construction={ef_construction})…")
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    store = np.lib.format.open_memmap(str(vec_path), mode="r")

    print(f"  Adding {n:,} vectors in chunks of {chunk_size:,}…")
    added = 0
    t0 = time.monotonic()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        index.add(np.array(store[start:end]))
        added = end
        elapsed = time.monotonic() - t0
        pct = 100.0 * added / n
        rate = added / elapsed if elapsed > 0 else 0
        print(
            f"    added {added:>12,} / {n:,}  ({pct:5.1f}%)  "
            f"{rate / 1e6:.2f}M vec/s",
            end="\r",
        )
    print()

    del store
    print(f"  Index built: {index.ntotal:,} vectors")
    return index


# ---------------------------------------------------------------------------
# Cache helpers  (meta formats match the benchmark scripts exactly)
# ---------------------------------------------------------------------------

def write_cache(
    cache_dir: Path,
    index: faiss.Index,
    queries: np.ndarray,
    n: int,
    dim: int,
    m: int,
    ef_construction: int,
    seed: int,
    index_type: str,  # "twophase" | "float32" | "int8"
    vec_src: Optional[Path] = None,  # if set, copy/link instead of expecting it already there
) -> None:
    """Write index.faiss, queries.npy, meta.json into cache_dir.

    vectors.npy is expected to already be in cache_dir (generated in-place),
    unless vec_src is provided, in which case it is copied.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # vectors.npy
    dst_vec = cache_dir / "vectors.npy"
    if vec_src is not None and vec_src != dst_vec:
        print(f"  Copying vectors.npy → {dst_vec}  ({vec_src.stat().st_size / (1024**3):.2f} GiB)…")
        shutil.copy2(str(vec_src), str(dst_vec))

    # index.faiss
    idx_path = cache_dir / "index.faiss"
    print(f"  Writing {idx_path}…")
    faiss.write_index(index, str(idx_path))

    # queries.npy
    qry_path = cache_dir / "queries.npy"
    np.save(str(qry_path), queries)

    # meta.json — format matches what each benchmark script expects
    meta_path = cache_dir / "meta.json"
    if index_type == "twophase":
        meta = {"n_vectors": n, "dim": dim, "m": m, "ef_construction": ef_construction, "seed": seed}
    else:
        meta = {"n_vectors": n, "dim": dim, "m": m, "ef_construction": ef_construction,
                "seed": seed, "index_type": index_type}
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Cache written: {cache_dir}")


def cache_is_valid(
    cache_dir: Path,
    n: int,
    dim: int,
    m: int,
    ef_construction: int,
    seed: int,
    index_type: str,
) -> bool:
    required = [
        cache_dir / "index.faiss",
        cache_dir / "vectors.npy",
        cache_dir / "queries.npy",
        cache_dir / "meta.json",
    ]
    if not all(p.exists() for p in required):
        return False
    try:
        with (cache_dir / "meta.json").open() as f:
            saved = json.load(f)
    except Exception:
        return False
    if index_type == "twophase":
        expected = {"n_vectors": n, "dim": dim, "m": m, "ef_construction": ef_construction, "seed": seed}
    else:
        expected = {"n_vectors": n, "dim": dim, "m": m, "ef_construction": ef_construction,
                    "seed": seed, "index_type": index_type}
    return saved == expected


# ---------------------------------------------------------------------------
# Per-type build routines
# ---------------------------------------------------------------------------

def build_twophase(args, n: int, rng: np.random.Generator, cache_dir: Path, queries: np.ndarray) -> None:
    print(f"\n{'═'*54}")
    print("  Building: two-phase (IndexHNSWSQ + float32 store)")
    print(f"{'═'*54}")

    if cache_is_valid(cache_dir, n, args.dim, args.m, args.ef_construction, args.seed, "twophase") and not args.rebuild:
        print(f"  Cache already valid — skipping build.  (Use --rebuild to force.)")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    vec_path = cache_dir / "vectors.npy"

    if vec_path.exists() and not args.rebuild:
        existing_shape = np.lib.format.open_memmap(str(vec_path), mode="r").shape
        if existing_shape == (n, args.dim):
            print(f"  vectors.npy already exists with correct shape — skipping generation.")
        else:
            print(f"  vectors.npy shape mismatch ({existing_shape} vs ({n}, {args.dim})) — regenerating.")
            generate_and_save_vectors_chunked(n, args.dim, rng, vec_path, args.build_chunk_size)
    else:
        generate_and_save_vectors_chunked(n, args.dim, rng, vec_path, args.build_chunk_size)

    index = build_hnsw_sq(vec_path, n, args.dim, args.m, args.ef_construction, args.ef_search, args.build_chunk_size)
    write_cache(cache_dir, index, queries, n, args.dim, args.m, args.ef_construction, args.seed, "twophase")


def build_float32(args, n: int, rng: np.random.Generator, cache_dir: Path, queries: np.ndarray,
                  shared_vec_path: Optional[Path] = None) -> None:
    print(f"\n{'═'*54}")
    print("  Building: float32 (IndexHNSWFlat)")
    print(f"{'═'*54}")

    if cache_is_valid(cache_dir, n, args.dim, args.m, args.ef_construction, args.seed, "float32") and not args.rebuild:
        print(f"  Cache already valid — skipping build.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    vec_path = cache_dir / "vectors.npy"

    if shared_vec_path is not None and shared_vec_path.exists():
        if vec_path.exists() and vec_path.stat().st_size == shared_vec_path.stat().st_size:
            print(f"  vectors.npy already present in {cache_dir} — skipping copy.")
        else:
            print(f"  Copying shared vectors.npy from {shared_vec_path}…")
            shutil.copy2(str(shared_vec_path), str(vec_path))
    elif not vec_path.exists():
        generate_and_save_vectors_chunked(n, args.dim, rng, vec_path, args.build_chunk_size)

    index = build_hnsw_flat(vec_path, n, args.dim, args.m, args.ef_construction, args.ef_search, args.build_chunk_size)
    write_cache(cache_dir, index, queries, n, args.dim, args.m, args.ef_construction, args.seed, "float32")


def build_int8(args, n: int, rng: np.random.Generator, cache_dir: Path, queries: np.ndarray,
               shared_vec_path: Optional[Path] = None) -> None:
    print(f"\n{'═'*54}")
    print("  Building: int8 (IndexHNSWSQ, no rerank store)")
    print(f"{'═'*54}")

    if cache_is_valid(cache_dir, n, args.dim, args.m, args.ef_construction, args.seed, "int8") and not args.rebuild:
        print(f"  Cache already valid — skipping build.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    vec_path = cache_dir / "vectors.npy"

    if shared_vec_path is not None and shared_vec_path.exists():
        if vec_path.exists() and vec_path.stat().st_size == shared_vec_path.stat().st_size:
            print(f"  vectors.npy already present in {cache_dir} — skipping copy.")
        else:
            print(f"  Copying shared vectors.npy from {shared_vec_path}…")
            shutil.copy2(str(shared_vec_path), str(vec_path))
    elif not vec_path.exists():
        generate_and_save_vectors_chunked(n, args.dim, rng, vec_path, args.build_chunk_size)

    index = build_hnsw_sq(vec_path, n, args.dim, args.m, args.ef_construction, args.ef_search, args.build_chunk_size)
    write_cache(cache_dir, index, queries, n, args.dim, args.m, args.ef_construction, args.seed, "int8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ── What to build ────────────────────────────────────────────────────
    parser.add_argument(
        "--type", choices=["twophase", "float32", "int8", "all"], default="all",
        help="Which index(es) to build (default: all)",
    )

    # ── Index / data parameters ──────────────────────────────────────────
    g = parser.add_argument_group("index / data")
    g.add_argument(
        "--total-index-gb", type=float, default=None,
        help="Memory budget in GiB.  N is derived from this using the per-type formula. "
             "Mutually exclusive with --n-vectors.",
    )
    g.add_argument(
        "--n-vectors", type=int, default=None,
        help="Explicit number of vectors (overrides --total-index-gb). "
             "Used as-is for all index types built in a single run.",
    )
    g.add_argument("--dim", type=int, default=128, help="Embedding dimension (default: 128)")
    g.add_argument("--m", type=int, default=16, help="HNSW M parameter (default: 16)")
    g.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction (default: 200)")
    g.add_argument("--ef-search", type=int, default=64, help="HNSW ef_search (default: 64)")
    g.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    g.add_argument("--query-pool-size", type=int, default=1000,
                   help="Number of query vectors to generate (default: 1000)")

    # ── Build performance ────────────────────────────────────────────────
    g2 = parser.add_argument_group("build performance")
    g2.add_argument(
        "--build-chunk-size", type=int, default=500_000,
        help="Vectors per chunk for generation and index.add() (default: 500000). "
             "Lower this if you are hitting RAM limits during the build.",
    )

    # ── Cache directories ────────────────────────────────────────────────
    g3 = parser.add_argument_group("cache directories")
    g3.add_argument("--twophase-cache-dir", default="index_cache",
                    help="Cache dir for the two-phase index (default: index_cache)")
    g3.add_argument("--float32-cache-dir", default="index_cache_float32",
                    help="Cache dir for the float32 baseline (default: index_cache_float32)")
    g3.add_argument("--int8-cache-dir", default="index_cache_int8",
                    help="Cache dir for the int8 baseline (default: index_cache_int8)")
    g3.add_argument("--rebuild", action="store_true",
                    help="Rebuild even if a valid cache already exists")

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────
    if args.total_index_gb is None and args.n_vectors is None:
        parser.error("Provide either --total-index-gb or --n-vectors")
    if args.total_index_gb is not None and args.n_vectors is not None:
        parser.error("--total-index-gb and --n-vectors are mutually exclusive")
    if args.build_chunk_size < 1000:
        parser.error("--build-chunk-size must be >= 1000")

    # ── Determine N for each type ─────────────────────────────────────────
    if args.n_vectors is not None:
        n_twophase = n_float32 = n_int8 = args.n_vectors
    else:
        n_twophase = compute_n_twophase(args.total_index_gb, args.dim, args.m)
        n_float32  = compute_n_float32(args.total_index_gb, args.dim, args.m)
        n_int8     = compute_n_int8(args.total_index_gb, args.dim, args.m)

    # For "all", use the two-phase N for all three so the dataset is shared
    if args.type == "all":
        n_float32 = n_int8 = n_twophase

    # ── Print budgets ─────────────────────────────────────────────────────
    types_to_build = ["twophase", "float32", "int8"] if args.type == "all" else [args.type]
    n_map = {"twophase": n_twophase, "float32": n_float32, "int8": n_int8}
    for t in types_to_build:
        print_budget(n_map[t], args.dim, args.m, t)

    # ── RNG (must be consumed in the same order regardless of --type) ─────
    rng = np.random.default_rng(args.seed)

    # ── Generate queries (small — always in RAM) ──────────────────────────
    print(f"Generating {args.query_pool_size} query vectors…")
    queries = generate_queries(args.query_pool_size, args.dim, rng)

    # ── Build ─────────────────────────────────────────────────────────────
    t_total = time.monotonic()

    if args.type == "twophase":
        build_twophase(args, n_twophase, rng, Path(args.twophase_cache_dir), queries)

    elif args.type == "float32":
        build_float32(args, n_float32, rng, Path(args.float32_cache_dir), queries)

    elif args.type == "int8":
        build_int8(args, n_int8, rng, Path(args.int8_cache_dir), queries)

    elif args.type == "all":
        # Build two-phase first (generates the shared vectors.npy)
        tp_dir = Path(args.twophase_cache_dir)
        build_twophase(args, n_twophase, rng, tp_dir, queries)

        shared_vec = tp_dir / "vectors.npy"

        # float32 and int8 reuse the same vectors — no re-generation
        build_float32(args, n_float32, rng, Path(args.float32_cache_dir), queries,
                      shared_vec_path=shared_vec)
        build_int8(args, n_int8, rng, Path(args.int8_cache_dir), queries,
                   shared_vec_path=shared_vec)

    elapsed = time.monotonic() - t_total
    print(f"\n{'═'*54}")
    print(f"  All done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'═'*54}")


if __name__ == "__main__":
    main()
