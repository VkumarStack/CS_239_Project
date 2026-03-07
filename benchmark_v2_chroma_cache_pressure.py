import argparse
import csv
import inspect
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
    li = int(rank)
    ui = min(li + 1, len(sorted_values) - 1)
    w = rank - li
    return sorted_values[li] * (1 - w) + sorted_values[ui] * w

def load_collection(client: chromadb.PersistentClient, collection_name: str | None):
    if collection_name:
        return client.get_collection(collection_name)

    collections = client.list_collections()
    if not collections:
        raise RuntimeError("No collections found in the persisted ChromaDB path")

    return client.get_collection(collections[0].name)


def fetch_random_query_embeddings(collection, query_count: int, random_dim: int | None = None) -> List[List[float]]:
    """
    Synthesize `query_count` random query embeddings and return as a list of lists.
    """
    # infer embedding dim if not provided
    if random_dim is None:
        try:
            result = collection.get(limit=1, offset=0, include=["embeddings"])
            embeddings = result.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                raise RuntimeError("Could not infer embedding dimension from collection")
            random_dim = len(embeddings[0])
        except Exception as e:
            raise RuntimeError("Failed to infer embedding dimension for random generation: " + str(e))

    print(f"Generating {query_count} random embeddings (dim={int(random_dim)})")
    rng = np.random.default_rng()
    arr = rng.random((query_count, int(random_dim)), dtype=np.float32)
    print(f"Generated {query_count} random embeddings with dim={int(random_dim)}")
    return arr.tolist()


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
    for emb in query_embeddings:
        start = time.perf_counter()
        if ef_mode == "query-argument":
            collection.query(query_embeddings=[emb], n_results=top_k, search_ef=ef_search)
        else:
            collection.query(query_embeddings=[emb], n_results=top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms

def warm_cache_with_vmtouch(data_path: str) -> None:
    if shutil.which("vmtouch") is None:
        raise RuntimeError("vmtouch not found. Install it or run without --preload-cache")
    print("Warming file cache via vmtouch...")
    # -t touch, -f follow symlinks, -q quiet
    cmd = ["vmtouch", "-t", "-f", "-q", data_path]
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"vmtouch failed with exit code {r.returncode}")


def vmtouch_residency_pct(data_path: str, timeout: float = 2.0) -> Optional[float]:
    """
    Returns % pages resident according to vmtouch, or None if vmtouch missing/fails.
    """
    if shutil.which("vmtouch") is None:
        print("vmtouch_residency_pct: vmtouch binary not found", file=sys.stderr)
        return None
    try:
        r = subprocess.run(["vmtouch", "-v", data_path], capture_output=True, text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"vmtouch_residency_pct: vmtouch timed out after {timeout}s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"vmtouch_residency_pct: vmtouch subprocess failed: {e}", file=sys.stderr)
        return None

    if r.returncode != 0:
        print(f"vmtouch_residency_pct: vmtouch returned rc={r.returncode} stderr={r.stderr}", file=sys.stderr)
        return None

    combined = (r.stdout or "") + "\n" + (r.stderr or "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)%", combined)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            print(f"vmtouch_residency_pct: regex parse error for '{m.group(1)}'", file=sys.stderr)
            return None

    print(f"vmtouch_residency_pct: parse failed, output:\n{combined}", file=sys.stderr)
    return None


def iter_data_files(root: str, min_size_bytes: int = 1) -> Iterable[Path]:
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if st.st_size >= min_size_bytes:
            yield p


def _page_size() -> int:
    return os.sysconf("SC_PAGE_SIZE")


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def fadvise_tree(root: str, advice: int, min_size_bytes: int = 1, verbose: bool = False) -> None:
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("os.posix_fadvise is unavailable on this platform")

    for p in iter_data_files(root, min_size_bytes=min_size_bytes):
        fd = os.open(p, os.O_RDONLY)
        try:
            st = os.fstat(fd)
            if st.st_size == 0:
                continue
            length = _align_up(st.st_size, _page_size())
            os.posix_fadvise(fd, 0, length, advice)
            if verbose:
                print(f"fadvise file={p} advice={advice} size={st.st_size} aligned_len={length}")
        finally:
            os.close(fd)


def warm_tree_by_read(root: str, min_size_bytes: int = 1, chunk_size: int = 8 * 1024 * 1024) -> None:
    for p in iter_data_files(root, min_size_bytes=min_size_bytes):
        with p.open("rb", buffering=0) as fh:
            while fh.read(chunk_size):
                pass


def maybe_direct_cache_transition(
    data_path: str,
    mode: Optional[str],
    min_size_bytes: int = 1,
    verbose: bool = False,
) -> None:
    if not mode or mode == "none":
        return
    if mode == "evict":
        fadvise_tree(data_path, os.POSIX_FADV_DONTNEED, min_size_bytes=min_size_bytes, verbose=verbose)
        return
    if mode == "willneed":
        fadvise_tree(data_path, os.POSIX_FADV_WILLNEED, min_size_bytes=min_size_bytes, verbose=verbose)
        return
    if mode == "readwarm":
        warm_tree_by_read(data_path, min_size_bytes=min_size_bytes)
        return
    raise ValueError(f"unknown direct cache mode: {mode}")


def read_meminfo_kb() -> dict:
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            parts = line.split()
            info[parts[0].rstrip(":")] = float(parts[1])  # in kB
    return info


def mem_total_kb() -> float:
    info = read_meminfo_kb()
    return float(info["MemTotal"])


def used_mem_pct() -> Optional[float]:
    """
    Used memory as (MemTotal - MemAvailable)/MemTotal * 100.
    This correlates with reclaim pressure better than Cached%.
    """
    try:
        info = read_meminfo_kb()
        mt = info.get("MemTotal")
        ma = info.get("MemAvailable")
        if not mt or not ma:
            return None
        used = mt - ma
        return (used / mt) * 100.0
    except Exception:
        return None


def mem_available_pct() -> Optional[float]:
    """
    MemAvailable/MemTotal * 100. As this drops, eviction pressure rises.
    """
    try:
        info = read_meminfo_kb()
        mt = info.get("MemTotal")
        ma = info.get("MemAvailable")
        if not mt or not ma:
            return None
        return (ma / mt) * 100.0
    except Exception:
        return None


def psi_memory() -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (some_avg10, full_avg10) from /proc/pressure/memory if available.
    """
    try:
        with open("/proc/pressure/memory") as fh:
            txt = fh.read().strip().splitlines()
        some = None
        full = None
        for line in txt:
            # example: some avg10=0.00 avg60=0.00 avg300=0.00 total=0
            if line.startswith("some "):
                some = float(line.split("avg10=")[1].split()[0])
            if line.startswith("full "):
                full = float(line.split("avg10=")[1].split()[0])
        return some, full
    except Exception:
        return None, None


# -----------------------------
# Stress control: purpose-built for eviction
# -----------------------------
def start_eviction_stress_vm(
    target_used_pct: Optional[int],
    target_bytes: Optional[int],
    vm_workers: int,
) -> Optional[subprocess.Popen]:
    """
    Start stress-ng that allocates anonymous memory and touches it (page-in),
    which forces eviction of file-backed cache (including Chroma's persisted pages).
    """
    if (not target_used_pct or target_used_pct <= 0) and (not target_bytes or target_bytes <= 0):
        return None

    if shutil.which("stress-ng") is None:
        raise RuntimeError("stress-ng not found. Install it before running pressure ramps")

    cmd = ["stress-ng", "--vm", str(vm_workers), "--vm-keep", "--page-in", "--metrics-brief", "--timeout", "3600s"]

    if target_bytes and target_bytes > 0:
        # distribute bytes evenly across workers
        per_worker = max(1, int(target_bytes) // max(1, vm_workers))
        cmd += ["--vm-bytes", f"{per_worker}B"]
    else:
        # This is % of total RAM used by each worker allocation pool (stress-ng interprets as percent of RAM)
        cmd += ["--vm-bytes", f"{int(target_used_pct)}%"]

    # IMPORTANT: capture stderr/stdout so we can detect OOM/early exits
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Fail fast if it exited immediately (very common with too-large allocations)
    time.sleep(0.25)
    if proc.poll() is not None:
        out, err = proc.communicate(timeout=1)
        raise RuntimeError(f"stress-ng exited early rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")

    return proc


def stop_stress_process(proc: Optional[subprocess.Popen]) -> None:
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


def parse_size(size_str: str) -> int:
    s = size_str.strip().upper()
    if not s:
        raise ValueError("empty size")

    multipliers = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

    if s[-1].isdigit():
        return int(s)

    unit = s[-1]
    if unit not in multipliers:
        raise ValueError(f"unknown size unit: {unit} in {size_str}")
    num = float(s[:-1])
    return int(num * multipliers[unit])


def parse_steps_bytes(steps_raw: str) -> List[int]:
    vals = [s.strip() for s in steps_raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("--mem-steps-bytes must have at least one value")
    return [parse_size(v) for v in vals]


def maybe_write_csv(csv_path: str | None, rows: List[dict]) -> None:
    if not csv_path:
        return
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ChromaDB query latency under ramping memory pressure designed to evict page cache."
    )
    parser.add_argument("--path", default="chroma_data", help="Path to persisted ChromaDB directory")
    parser.add_argument("--collection", default=None, help="Collection name (default: first collection)")
    parser.add_argument("--queries-per-step", type=int, default=200, help="Queries executed at each pressure step")
    parser.add_argument("--top-k", type=int, default=10, help="n_results per query")
    parser.add_argument("--ef-search", type=int, default=100, help="HNSW ef_search value")

    # Steps: prefer percent-of-RAM used by stress, since it scales to VM size.
    parser.add_argument(
        "--mem-steps",
        default="0,40,60,75,85,90",
        help="Comma-separated target USED memory percentages (via stress-ng --vm-bytes %%).",
    )
    parser.add_argument(
        "--mem-steps-bytes",
        default=None,
        help="Comma-separated absolute sizes (e.g. 1G,2G). Overrides --mem-steps when provided.",
    )
    parser.add_argument("--vm-workers", type=int, default=2, help="stress-ng --vm worker count")
    parser.add_argument("--settle-seconds", type=float, default=3.0, help="Wait time after starting stress before measuring")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--query-mode",
        choices=["resample", "fixed", "presampled"],
        default="fixed",
        help="fixed is recommended so latency changes reflect cache pressure, not query mix variance.",
    )
    parser.add_argument("--preload-cache", action="store_true", help="Warm cache with vmtouch before test")
    parser.add_argument("--track-vmtouch", action="store_true", help="Record vmtouch residency % each step (requires vmtouch)")
    parser.add_argument(
        "--direct-cache-mode",
        choices=["none", "evict", "willneed", "readwarm"],
        default="none",
        help="Apply direct file-cache control to persisted files before benchmark or each step.",
    )
    parser.add_argument(
        "--direct-cache-before-benchmark",
        action="store_true",
        help="Apply direct cache transition once before the benchmark starts.",
    )
    parser.add_argument(
        "--direct-cache-before-each-step",
        action="store_true",
        help="Apply direct cache transition before each pressure step.",
    )
    parser.add_argument(
        "--direct-cache-min-size-bytes",
        type=int,
        default=4096,
        help="Ignore tiny files when applying direct cache control.",
    )
    parser.add_argument(
        "--direct-cache-verbose",
        action="store_true",
        help="Print per-file direct cache operations.",
    )
    parser.add_argument("--csv-out", default=None, help="Optional path to write per-step CSV results")
    parser.add_argument(
        "--timeline-out",
        default=None,
        help="Optional per-query timeline CSV (elapsed_sec, latency_ms, target_used_pct, used_pct, mem_avail_pct, psi_some10, psi_full10, vmtouch_residency_pct).",
    )

    args = parser.parse_args()

    if args.queries_per_step < 1:
        raise ValueError("--queries-per-step must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.ef_search < 1:
        raise ValueError("--ef-search must be >= 1")

    if args.mem_steps_bytes:
        mem_steps = parse_steps_bytes(args.mem_steps_bytes)
        mem_is_bytes = True
    else:
        mem_steps = parse_steps(args.mem_steps)
        mem_is_bytes = False

    random.seed(args.seed)
    np.random.seed(args.seed)

    mt_kb = mem_total_kb()
    print(f"Detected MemTotal: {mt_kb/1024/1024:.2f} GiB")

    def bytes_to_used_pct(b: int) -> Optional[float]:
        if not b:
            return None
        return (b / 1024.0) / mt_kb * 100.0

    print(f"Opening persisted ChromaDB at: {args.path}")
    client = chromadb.PersistentClient(path=args.path)
    collection = load_collection(client, args.collection)
    print(f"Using collection: {collection.name}")
    print(f"Collection vector count: {collection.count()}")

    if args.preload_cache:
        warm_cache_with_vmtouch(args.path)

    ef_mode = apply_ef_search(collection, args.ef_search)
    print(f"ef_search mode: {ef_mode}")

    if args.direct_cache_before_benchmark and args.direct_cache_mode != "none":
        print(f"Applying direct cache mode before benchmark: {args.direct_cache_mode}")
        maybe_direct_cache_transition(
            data_path=args.path,
            mode=args.direct_cache_mode,
            min_size_bytes=args.direct_cache_min_size_bytes,
            verbose=args.direct_cache_verbose,
        )

    fixed_query_embeddings = None
    presampled_query_sets: List[List[List[float]]] = []

    if args.query_mode == "fixed":
        print("Preparing one fixed random query set (generated) ...")
        fixed_query_embeddings = fetch_random_query_embeddings(collection, args.queries_per_step)
    elif args.query_mode == "presampled":
        print("Pre-sampling random query sets for all steps...")
        for _ in mem_steps:
            presampled_query_sets.append(fetch_random_query_embeddings(collection, args.queries_per_step))

    print("\n=== Eviction Pressure Ramp Benchmark ===")
    pretty = ",".join([f"{v}B" for v in mem_steps]) if mem_is_bytes else ",".join([str(v) for v in mem_steps])
    print(f"steps: {pretty}  (bytes)" if mem_is_bytes else f"steps: {pretty}  (target USED %)")
    print(f"queries/step: {args.queries_per_step} | query-mode: {args.query_mode} | top_k: {args.top_k} | vm-workers: {args.vm_workers}")

    rows: List[dict] = []
    timeline_rows: List[dict] = []

    bench_start = time.time()

    for step_idx, raw_step in enumerate(mem_steps, start=1):
        stress_proc = None
        try:
            if args.direct_cache_before_each_step and args.direct_cache_mode != "none":
                print(f"Applying direct cache mode before step {step_idx}: {args.direct_cache_mode}")
                maybe_direct_cache_transition(
                    data_path=args.path,
                    mode=args.direct_cache_mode,
                    min_size_bytes=args.direct_cache_min_size_bytes,
                    verbose=args.direct_cache_verbose,
                )

            if mem_is_bytes:
                target_used_pct = bytes_to_used_pct(raw_step)
                stress_proc = start_eviction_stress_vm(None, raw_step, args.vm_workers)
                target_label = f"{raw_step}B (~{target_used_pct:.1f}%)" if target_used_pct is not None else f"{raw_step}B"
            else:
                target_used_pct = float(raw_step)
                stress_proc = start_eviction_stress_vm(int(raw_step), None, args.vm_workers)
                target_label = f"{raw_step}%"

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

            # Step-level telemetry (sample once pre-queries)
            used_pct_now = used_mem_pct()
            avail_pct_now = mem_available_pct()
            psi_some10, psi_full10 = psi_memory()
            vm_res_pct = vmtouch_residency_pct(args.path) if args.track_vmtouch else None

            latencies_ms = run_queries(
                collection=collection,
                query_embeddings=query_embeddings,
                top_k=args.top_k,
                ef_mode=ef_mode,
                ef_search=args.ef_search,
            )

            # per-query timeline telemetry (sample lightweight metrics)
            for lat in latencies_ms:
                elapsed = time.time() - bench_start
                u = used_mem_pct()
                a = mem_available_pct()
                ps, pf = psi_memory()
                vt = vmtouch_residency_pct(args.path) if args.track_vmtouch else None
                timeline_rows.append(
                    {
                        "elapsed_sec": round(elapsed, 6),
                        "latency_ms": round(float(lat), 6),
                        "target_used_pct": round(float(target_used_pct), 6) if target_used_pct is not None else None,
                        "used_pct": round(float(u), 6) if u is not None else None,
                        "mem_avail_pct": round(float(a), 6) if a is not None else None,
                        "psi_some_avg10": round(float(ps), 6) if ps is not None else None,
                        "psi_full_avg10": round(float(pf), 6) if pf is not None else None,
                        "vmtouch_residency_pct": round(float(vt), 6) if vt is not None else None,
                    }
                )

            sorted_lat = sorted(latencies_ms)
            row = {
                "step_index": step_idx,
                "target_used_pct": raw_step if not mem_is_bytes else round(float(target_used_pct), 3) if target_used_pct else None,
                "target_bytes": raw_step if mem_is_bytes else None,
                "queries": len(sorted_lat),
                "query_mode": args.query_mode,
                "top_k": args.top_k,
                "ef_search": args.ef_search,
                "cache_preloaded": args.preload_cache,
                "direct_cache_mode": args.direct_cache_mode,
                "direct_cache_before_benchmark": args.direct_cache_before_benchmark,
                "direct_cache_before_each_step": args.direct_cache_before_each_step,
                # Step telemetry sampled once:
                "used_pct_at_step_start": round(float(used_pct_now), 3) if used_pct_now is not None else None,
                "mem_avail_pct_at_step_start": round(float(avail_pct_now), 3) if avail_pct_now is not None else None,
                "psi_some_avg10_at_step_start": round(float(psi_some10), 6) if psi_some10 is not None else None,
                "psi_full_avg10_at_step_start": round(float(psi_full10), 6) if psi_full10 is not None else None,
                "vmtouch_residency_pct_at_step_start": round(float(vm_res_pct), 3) if vm_res_pct is not None else None,
                # Latency stats:
                "avg_ms": round(statistics.mean(sorted_lat), 3),
                "p50_ms": round(percentile(sorted_lat, 50), 3),
                "p99_ms": round(percentile(sorted_lat, 99), 3),
                "min_ms": round(sorted_lat[0], 3),
                "max_ms": round(sorted_lat[-1], 3),
            }
            rows.append(row)

            print(
                f"step {step_idx}/{len(mem_steps)} | target={target_label} | "
                f"used={row['used_pct_at_step_start']}% | avail={row['mem_avail_pct_at_step_start']}% | "
                f"p50={row['p50_ms']} ms | p99={row['p99_ms']} ms | "
                f"psi_some10={row['psi_some_avg10_at_step_start']} | psi_full10={row['psi_full_avg10_at_step_start']} | "
                f"vmtouch%={row['vmtouch_residency_pct_at_step_start']}"
            )

        finally:
            stop_stress_process(stress_proc)

    # Write per-step CSV
    if args.csv_out:
        maybe_write_csv(args.csv_out, rows)
        print(f"\nWrote CSV: {args.csv_out}")

    # Write per-query timeline
    if args.timeline_out:
        outp = Path(args.timeline_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "elapsed_sec",
            "latency_ms",
            "target_pressure_pct",
            "actual_pressure_pct",
        ]
        with outp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in timeline_rows:
                w.writerow(
                    {
                        "elapsed_sec": ("" if r.get("elapsed_sec") is None else r.get("elapsed_sec")),
                        "latency_ms": ("" if r.get("latency_ms") is None else r.get("latency_ms")),
                        # map internal names to the plotting schema
                        "target_pressure_pct": ("" if r.get("target_used_pct") is None else r.get("target_used_pct")),
                        "actual_pressure_pct": ("" if r.get("used_pct") is None else r.get("used_pct")),
                    }
                )
        print(f"Wrote timeline CSV: {args.timeline_out}")


if __name__ == "__main__":
    main()
