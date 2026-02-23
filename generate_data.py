import numpy as np
import chromadb
import time

# --- CONFIGURATION ---
NUM_VECTORS = 250_000     # Adjust this to hit your 4GB target
DIMENSIONS = 960          # Matches GIST-1M dataset size
BATCH_SIZE = 10_000       # Efficient batch size for ingestion
HOST_PORT = 8001          # Must match the port in docker-compose.yaml

def main():
    print(f"Connecting to ChromaDB on localhost:{HOST_PORT}...")
    
    # Connect to the Docker container via HTTP
    # Note: We use HttpClient because the DB is in a separate container
    client = chromadb.HttpClient(host='localhost', port=HOST_PORT)
    
    # 1. Dynamically get the limit from the server
    max_batch = client.get_max_batch_size()
    print(f"Server max batch size: {max_batch}")

    # 2. Set your batch size to be safe (use the limit)
    BATCH_SIZE = max_batch

    # Create the collection (or get it if it exists)
    # metadata={"hnsw:space": "cosine"} optimizes for cosine similarity
    collection = client.get_or_create_collection(
        name="noise_test", 
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Target Memory Size: {NUM_VECTORS * DIMENSIONS * 4 / 1024**3:.2f} GB (Raw Data Only)")
    print("Starting data generation and insertion...")
    
    start_total = time.time()
    
    for i in range(0, NUM_VECTORS, BATCH_SIZE):
        # 1. Generate random vectors (Float32)
        embeddings = np.random.rand(BATCH_SIZE, DIMENSIONS).astype(np.float32)
        
        # 2. Generate simple string IDs
        ids = [str(x) for x in range(i, i + BATCH_SIZE)]
        
        # 3. Insert into ChromaDB
        collection.add(embeddings=embeddings, ids=ids)
        
        # Optional: Print progress every 10 batches
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"Inserted batch {i} to {i+BATCH_SIZE}...")

    print(f"Finished! Total vectors in collection: {collection.count()}")
    print(f"Total time taken: {time.time() - start_total:.2f} seconds")

if __name__ == "__main__":
    main()