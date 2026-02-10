import chromadb
import time
import numpy as np
import subprocess
import sys
import re

# --- CONFIGURATION ---
HOST_PORT = 8001
COLLECTION_NAME = "noise_test"
DIMENSIONS = 960 
WARMUP_QUERIES = 20
BENCHMARK_QUERIES = 200
CONTAINER_NAME = "chromadb_research"
DATA_DIR = "/data"

# --- TOGGLE ORDER ---
# Set to True to run NOISE first, then BASELINE
# Set to False for standard BASELINE -> NOISE
RUN_NOISY_FIRST = True  

# Global variable for stress command
STRESS_CMD = []
KILL_CMD = ["docker", "exec", CONTAINER_NAME, "pkill", "stress-ng"]

def run_docker_cmd(cmd_list, check=True):
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Docker Error: {e.stderr}")
        if check: sys.exit(1)
        return None

def setup_environment():
    """Installs tools and Calculates Stress Size."""
    print("\n--- 1. ENVIRONMENT ANALYSIS ---")
    print("Checking tools...", end=" ")
    run_docker_cmd(["docker", "exec", "-u", "0", CONTAINER_NAME, "bash", "-c", 
                   "apt-get update -qq && apt-get install -y -qq stress-ng vmtouch procps bc"])
    print("Done.")

    # Get RAM and Dataset Size
    mem_info = run_docker_cmd(["docker", "exec", CONTAINER_NAME, "grep", "MemTotal", "/proc/meminfo"])
    total_ram_kb = int(re.search(r"(\d+)", mem_info).group(1))
    total_ram_gb = total_ram_kb / 1024 / 1024
    
    du_output = run_docker_cmd(["docker", "exec", CONTAINER_NAME, "du", "-sb", DATA_DIR])
    dataset_bytes = int(du_output.split()[0])
    dataset_gb = dataset_bytes / (1024**3)

    print(f"Container RAM: {total_ram_gb:.2f} GB")
    print(f"Dataset Size:  {dataset_gb:.2f} GB")

    if dataset_gb > (total_ram_gb * 0.9):
        print("\n[CRITICAL WARNING] Dataset is too large!")
        stress_size = "256M"
    else:
        free_space_gb = total_ram_gb - dataset_gb
        target_stress_gb = free_space_gb + 0.2
        stress_size = f"{int(target_stress_gb * 1024)}M"
        print(f"Target Stress: {stress_size} (Calculated to saturate memory)")

    global STRESS_CMD
    STRESS_CMD = [
        "docker", "exec", "-d", CONTAINER_NAME, 
        "stress-ng", 
        "--vm", "2", 
        "--vm-bytes", stress_size, 
        "--timeout", "60s"
    ]

def force_load_data():
    """Forces 100% Cache Residency."""
    print(f"\n[System] Re-Warming Cache (vmtouch {DATA_DIR})...")
    try:
        run_docker_cmd(["docker", "exec", CONTAINER_NAME, "vmtouch", "-t", DATA_DIR])
        residency = run_docker_cmd(["docker", "exec", CONTAINER_NAME, "vmtouch", DATA_DIR])
        match = re.search(r"Resident Pages:.*?\s+(\d+)%", residency)
        if match:
            percent = int(match.group(1))
            print(f"RAM Residency: {percent}%")
    except Exception as e:
        print(f"Warmup failed: {e}")

def get_latencies(collection, n_queries, label):
    print(f"\n--- {label} ({n_queries} queries) ---")
    latencies = []
    queries = np.random.rand(n_queries, DIMENSIONS).astype(np.float32).tolist()

    for i, q in enumerate(queries):
        start = time.perf_counter()
        collection.query(query_embeddings=[q], n_results=10)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        
        if (i+1) % 50 == 0:
            sys.stdout.write(f"\rProgress: {i+1}/{n_queries}")
            sys.stdout.flush()
    return latencies

def print_stats(latencies, label):
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    print(f"\n{label}: P50={p50:.2f}ms | P99={p99:.2f}ms")
    return p99

def run_baseline(collection):
    # Ensure clean state before running
    force_load_data()
    lats = get_latencies(collection, BENCHMARK_QUERIES, "BASELINE")
    return print_stats(lats, "BASELINE")

def run_stress(collection):
    # Ensure clean state before running
    force_load_data()
    print(f"\n\n>>> INJECTING NOISE: {' '.join(STRESS_CMD[4:])} <<<")
    subprocess.run(STRESS_CMD, check=True)
    print("Waiting 5s for memory saturation...")
    time.sleep(5) 
    
    lats = get_latencies(collection, BENCHMARK_QUERIES, "STRESSED")
    p99 = print_stats(lats, "STRESSED")
    
    # Clean up noise immediately
    subprocess.run(KILL_CMD)
    return p99

def main():
    try:
        setup_environment()

        print(f"\nConnecting to ChromaDB on localhost:{HOST_PORT}...")
        client = chromadb.HttpClient(host='localhost', port=HOST_PORT)
        collection = client.get_collection(name=COLLECTION_NAME)

        base_p99 = 0
        stress_p99 = 0

        if RUN_NOISY_FIRST:
            print("\n[ORDER] Running STRESSED Test FIRST.")
            stress_p99 = run_stress(collection)
            
            print("\n[ORDER] profound Cooling Down (5s)...")
            time.sleep(5)
            
            print("\n[ORDER] Running BASELINE Test SECOND.")
            base_p99 = run_baseline(collection)
        else:
            print("\n[ORDER] Running BASELINE Test FIRST.")
            base_p99 = run_baseline(collection)
            
            print("\n[ORDER] Running STRESSED Test SECOND.")
            stress_p99 = run_stress(collection)

        # Final Report
        degradation = ((stress_p99 - base_p99) / base_p99) * 100
        print("\n" + "="*40)
        print(f"Order: {'Stressed -> Baseline' if RUN_NOISY_FIRST else 'Baseline -> Stressed'}")
        print(f"Baseline P99: {base_p99:.2f} ms")
        print(f"Stressed P99: {stress_p99:.2f} ms")
        print(f"Degradation:  {degradation:.1f}%")
        print("="*40)

    except KeyboardInterrupt:
        print("\nStopping...")
        subprocess.run(KILL_CMD)

if __name__ == "__main__":
    main()