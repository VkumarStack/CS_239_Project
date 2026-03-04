import numpy as np
import chromadb
import time

# --- CONFIGURATION ---
NUM_VECTORS = 35_791_394  # ~128GB target (128 * 1024^3 / (960 * 4 bytes))
DIMENSIONS = 960          # Matches GIST-1M dataset size
BATCH_SIZE = 10_000       # Efficient batch size for ingestion
DATA_PATH = "chroma_data" # Local directory for ChromaDB data

def main():
    print(f"Using local ChromaDB at: {DATA_PATH}")
    
    # Use PersistentClient for local storage
    client = chromadb.PersistentClient(path=DATA_PATH)

    # Dynamically get the limit from the client
    max_batch = client.get_max_batch_size()
    print(f"Client max batch size: {max_batch}")
    
    # Use the actual max batch size
    actual_batch_size = max_batch
    
    # Create the collection (or get it if it exists)
    # metadata={"hnsw:space": "cosine"} optimizes for cosine similarity
    collection = client.get_or_create_collection(
        name="noise_test", 
        metadata={"hnsw:space": "cosine"}
    )
    
    # Resume from existing data
    already_inserted = collection.count()
    print(f"Already inserted: {already_inserted} vectors ({already_inserted * DIMENSIONS * 4 / 1024**3:.2f} GB)")
    print(f"Target Memory Size: {NUM_VECTORS * DIMENSIONS * 4 / 1024**3:.2f} GB (Raw Data Only)")
    print(f"Remaining: {(NUM_VECTORS - already_inserted) * DIMENSIONS * 4 / 1024**3:.2f} GB")
    print("Starting data generation and insertion...")
    
    start_total = time.time()
    
    for i in range(already_inserted, NUM_VECTORS, actual_batch_size):
        # Calculate remaining vectors to avoid exceeding NUM_VECTORS
        remaining = NUM_VECTORS - i
        current_batch = min(actual_batch_size, remaining)
        
        # 1. Generate random vectors (Float32)
        embeddings = np.random.rand(current_batch, DIMENSIONS).astype(np.float32)
        
        # 2. Generate simple string IDs
        ids = [str(x) for x in range(i, i + current_batch)]
        
        # 3. Insert into ChromaDB
        collection.add(embeddings=embeddings, ids=ids)
        
        # Optional: Print progress every 10 batches
        if (i // actual_batch_size) % 10 == 0:
            print(f"Inserted batch {i} to {i+current_batch}...")

    print(f"Finished! Total vectors in collection: {collection.count()}")
    print(f"Total time taken: {time.time() - start_total:.2f} seconds")

if __name__ == "__main__":
    main()