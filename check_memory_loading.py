import chromadb
import psutil
import os
import time
import random

def get_process_memory_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_system_memory_info():
    """Get system memory info"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'available_gb': mem.available / (1024**3),
        'used_gb': mem.used / (1024**3),
        'percent': mem.percent,
        'cached_gb': getattr(mem, 'cached', 0) / (1024**3)
    }

def main():
    print("=== ChromaDB Memory Loading Behavior Test ===\n")
    
    # Step 1: Baseline memory
    print("Step 1: Baseline Memory")
    baseline_process = get_process_memory_mb()
    baseline_system = get_system_memory_info()
    print(f"  Process Memory: {baseline_process:.2f} MB")
    print(f"  System Used: {baseline_system['used_gb']:.2f} GB / {baseline_system['total_gb']:.2f} GB ({baseline_system['percent']:.1f}%)")
    print(f"  System Cached: {baseline_system['cached_gb']:.2f} GB\n")
    
    time.sleep(2)
    
    # Step 2: Open ChromaDB client (no collection access yet)
    print("Step 2: Opening ChromaDB Client")
    client = chromadb.PersistentClient(path="chroma_data")
    after_client_process = get_process_memory_mb()
    after_client_system = get_system_memory_info()
    print(f"  Process Memory: {after_client_process:.2f} MB (Δ +{after_client_process - baseline_process:.2f} MB)")
    print(f"  System Used: {after_client_system['used_gb']:.2f} GB (Δ +{after_client_system['used_gb'] - baseline_system['used_gb']:.2f} GB)")
    print(f"  System Cached: {after_client_system['cached_gb']:.2f} GB (Δ +{after_client_system['cached_gb'] - baseline_system['cached_gb']:.2f} GB)\n")
    
    time.sleep(2)
    
    # Step 3: Get collection (may trigger index loading)
    print("Step 3: Loading Collection")
    collection = client.get_collection("noise_test")
    total_vectors = collection.count()
    print(f"  Collection has {total_vectors:,} vectors")
    
    time.sleep(2)
    
    after_collection_process = get_process_memory_mb()
    after_collection_system = get_system_memory_info()
    print(f"  Process Memory: {after_collection_process:.2f} MB (Δ +{after_collection_process - after_client_process:.2f} MB)")
    print(f"  System Used: {after_collection_system['used_gb']:.2f} GB (Δ +{after_collection_system['used_gb'] - after_client_system['used_gb']:.2f} GB)")
    print(f"  System Cached: {after_collection_system['cached_gb']:.2f} GB (Δ +{after_collection_system['cached_gb'] - baseline_system['cached_gb']:.2f} GB total)\n")
    
    # Step 4: Run a few queries to see if memory increases
    print("Step 4: Running 10 Queries")
    for i in range(10):
        # Get a random vector from collection
        offset = random.randint(0, total_vectors - 1)
        result = collection.get(limit=1, offset=offset, include=["embeddings"])
        query_embedding = result["embeddings"][0]
        
        # Query
        collection.query(query_embeddings=[query_embedding], n_results=10)
        
        if i % 3 == 0:
            current_process = get_process_memory_mb()
            current_system = get_system_memory_info()
            print(f"  After query {i+1}: Process={current_process:.2f} MB, System Used={current_system['used_gb']:.2f} GB")
    
    time.sleep(2)
    
    after_queries_process = get_process_memory_mb()
    after_queries_system = get_system_memory_info()
    print(f"\n  Final Process Memory: {after_queries_process:.2f} MB (Δ +{after_queries_process - after_collection_process:.2f} MB since collection load)")
    print(f"  Final System Used: {after_queries_system['used_gb']:.2f} GB (Δ +{after_queries_system['used_gb'] - after_collection_system['used_gb']:.2f} GB)")
    print(f"  Final System Cached: {after_queries_system['cached_gb']:.2f} GB (Δ +{after_queries_system['cached_gb'] - baseline_system['cached_gb']:.2f} GB total)\n")
    
    # Summary
    print("=== Summary ===")
    print(f"Total process memory increase: {after_queries_process - baseline_process:.2f} MB")
    print(f"Total system memory increase: {(after_queries_system['used_gb'] - baseline_system['used_gb']) * 1024:.2f} MB")
    print(f"Total system cache increase: {(after_queries_system['cached_gb'] - baseline_system['cached_gb']) * 1024:.2f} MB")
    print("\nInterpretation:")
    print("- Large memory increase at Step 3 → ChromaDB loads index into memory immediately")
    print("- Small increase at Step 3, gradual during queries → ChromaDB loads lazily (page by page)")
    print("- Large cache increase → Data is in page cache (can be evicted by OS)")

if __name__ == "__main__":
    main()
