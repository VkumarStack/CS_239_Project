import argparse
import inspect
import random
import statistics
import time
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

    first_collection = collections[0]
    return client.get_collection(first_collection.name)


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


def run_benchmark(collection, query_embeddings: List[List[float]], top_k: int) -> List[float]:
    latencies_ms: List[float] = []

    for embedding in query_embeddings:
        start = time.perf_counter()
        collection.query(query_embeddings=[embedding], n_results=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    return latencies_ms


def run_warmup(collection, query_embeddings: List[List[float]], top_k: int) -> None:
    for embedding in query_embeddings:
        collection.query(query_embeddings=[embedding], n_results=top_k)


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


def run_benchmark_with_ef(
    collection,
    query_embeddings: List[List[float]],
    top_k: int,
    ef_search: int,
    ef_mode: str,
) -> List[float]:
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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark single-client ChromaDB query latency (P50/P99) using persisted local data"
    )
    parser.add_argument("--path", default="chroma_data", help="Path to persisted ChromaDB directory")
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to benchmark (default: first collection found)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=200,
        help="Number of random query vectors to benchmark",
    )
    parser.add_argument("--top-k", type=int, default=10, help="n_results for each query")
    parser.add_argument(
        "--ef-search",
        type=int,
        default=100,
        help="HNSW ef_search value (Chroma default is commonly 100)",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["warm", "cold"],
        default="cold",
        help="Cache mode: warm runs an untimed warmup pass before measurement; cold measures first pass",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Opening persisted ChromaDB at: {args.path}")
    client = chromadb.PersistentClient(path=args.path)

    collection = load_collection(client, args.collection)
    print(f"Using collection: {collection.name}")
    print(f"Collection vector count: {collection.count()}")

    if args.ef_search < 1:
        raise ValueError("--ef-search must be >= 1")

    ef_mode = apply_ef_search(collection, args.ef_search)

    print("Preparing random query embeddings from existing vectors...")
    query_embeddings = fetch_random_query_embeddings(collection, args.queries)

    if args.cache_mode == "warm":
        print("Running warmup pass to warm caches (untimed)...")
        if ef_mode == "query-argument":
            run_benchmark_with_ef(
                collection,
                query_embeddings,
                args.top_k,
                args.ef_search,
                ef_mode,
            )
        else:
            run_warmup(collection, query_embeddings, args.top_k)

    print("Running sequential benchmark (no stressors / single client)...")
    latencies_ms = run_benchmark_with_ef(
        collection,
        query_embeddings,
        args.top_k,
        args.ef_search,
        ef_mode,
    )

    sorted_latencies = sorted(latencies_ms)
    p50 = percentile(sorted_latencies, 50)
    p99 = percentile(sorted_latencies, 99)
    avg = statistics.mean(sorted_latencies)

    print("\n=== ChromaDB Query Latency Results ===")
    print(f"Queries executed: {len(sorted_latencies)}")
    print(f"top_k (n_results): {args.top_k}")
    print(f"ef_search: {args.ef_search} (applied via {ef_mode})")
    print(f"cache_mode: {args.cache_mode}")
    print(f"Average: {avg:.3f} ms")
    print(f"P50: {p50:.3f} ms")
    print(f"P99: {p99:.3f} ms")
    print(f"Min: {sorted_latencies[0]:.3f} ms")
    print(f"Max: {sorted_latencies[-1]:.3f} ms")


if __name__ == "__main__":
    main()
