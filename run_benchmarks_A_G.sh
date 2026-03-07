#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCH_SCRIPT="${BENCH_SCRIPT:-benchmark_v2_chroma_cache_continuous_adaptive.py}"
CHROMA_PATH="${CHROMA_PATH:-chroma_data}"
OUT_DIR="${OUT_DIR:-outputs}"
REPEATS="${REPEATS:-1}"

BASE_DURATION="${BASE_DURATION:-300}"
STRESS_DURATION="${STRESS_DURATION:-400}"
TRACK_VMTOUCH="${TRACK_VMTOUCH:-1}"
SAMPLE_VMTOUCH_EVERY_N_QUERIES="${SAMPLE_VMTOUCH_EVERY_N_QUERIES:-50}"
SAMPLE_USED_MEM_EVERY_N_QUERIES="${SAMPLE_USED_MEM_EVERY_N_QUERIES:-10}"
PRELOAD_CACHE="${PRELOAD_CACHE:-0}"
VMTOUCH_TIMEOUT_SECONDS="${VMTOUCH_TIMEOUT_SECONDS:-2.0}"

mkdir -p "$OUT_DIR"

echo "Runner config:"
echo "  CHROMA_PATH=${CHROMA_PATH}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  REPEATS=${REPEATS}"
echo "  TRACK_VMTOUCH=${TRACK_VMTOUCH} (sample every ${SAMPLE_VMTOUCH_EVERY_N_QUERIES} queries)"
echo "  PRELOAD_CACHE=${PRELOAD_CACHE}"
echo "  VMTOUCH_TIMEOUT_SECONDS=${VMTOUCH_TIMEOUT_SECONDS}"

run_profile() {
  local name="$1"
  local duration="$2"
  shift 2

  local rep
  for ((rep = 1; rep <= REPEATS; rep++)); do
    local suffix=""
    if [[ "$REPEATS" -gt 1 ]]; then
      suffix="_r${rep}"
    fi

    local csv_out="${OUT_DIR}/v2_${name}${suffix}.csv"
    local timeline_out="${OUT_DIR}/v2_${name}${suffix}_timeline.csv"

    echo ""
    echo "=== Running ${name}${suffix} ==="
    echo "CSV: ${csv_out}"

    cmd=(
      "$PYTHON_BIN" "$BENCH_SCRIPT"
      --path "$CHROMA_PATH" \
      --duration-seconds "$duration" \
      --query-pool-size 2000 \
      --query-interval-ms 0 \
      --eval-window-queries 300 \
      --min-eval-samples 150 \
      --controller-cooldown-queries 150 \
      --auto-latency-thresholds \
      --auto-threshold-warmup-queries 150 \
      --auto-threshold-calm-multiplier 0.95 \
      --auto-threshold-spike-multiplier 1.20 \
      --sample-used-mem-every-n-queries "$SAMPLE_USED_MEM_EVERY_N_QUERIES" \
      --vmtouch-timeout-seconds "$VMTOUCH_TIMEOUT_SECONDS" \
      --ef-mode collection-default \
      --csv-out "$csv_out" \
      --timeline-out "$timeline_out" \
      "$@"
    )

    if [[ "$TRACK_VMTOUCH" == "1" ]]; then
      cmd+=(--track-vmtouch --sample-vmtouch-every-n-queries "$SAMPLE_VMTOUCH_EVERY_N_QUERIES")
    fi
    if [[ "$PRELOAD_CACHE" == "1" ]]; then
      cmd+=(--preload-cache)
    fi

    "${cmd[@]}"
  done
}

# A: baseline (no stress, no direct cache forcing)
run_profile "A_baseline" "$BASE_DURATION" \
  --disable-stress \
  --direct-cache-mode none \
  --max-pressure-pct 0 \
  --top-k 10 \
  --ef-search 100

# B: cache-only cold-shock (evict)
run_profile "B_cache_evict" "$BASE_DURATION" \
  --disable-stress \
  --direct-cache-mode evict \
  --direct-cache-on-start \
  --direct-cache-every-n-queries 100 \
  --top-k 10 \
  --ef-search 100

# C: cache-only warm hint (willneed)
run_profile "C_cache_willneed" "$BASE_DURATION" \
  --disable-stress \
  --direct-cache-mode willneed \
  --direct-cache-on-start \
  --direct-cache-every-n-queries 100 \
  --top-k 10 \
  --ef-search 100

# D: real pressure only (stress, no direct cache forcing)
run_profile "D_stress_only" "$STRESS_DURATION" \
  --initial-pressure-pct 5 \
  --min-pressure-pct 0 \
  --max-pressure-pct 85 \
  --pressure-step-pct 5 \
  --vm-workers 2 \
  --total-used-pct \
  --direct-cache-mode none \
  --pressure-spike-threshold-pct 92 \
  --pressure-calm-threshold-pct 72 \
  --top-k 10 \
  --ef-search 100

# E: combined worst-case (stress + on-change evict)
run_profile "E_stress_plus_evict" "$STRESS_DURATION" \
  --initial-pressure-pct 5 \
  --min-pressure-pct 0 \
  --max-pressure-pct 85 \
  --pressure-step-pct 5 \
  --vm-workers 2 \
  --total-used-pct \
  --direct-cache-mode evict \
  --direct-cache-on-change \
  --pressure-spike-threshold-pct 92 \
  --pressure-calm-threshold-pct 72 \
  --top-k 10 \
  --ef-search 100

# F: workload sensitivity (heavier retrieval)
run_profile "F_heavy" "$STRESS_DURATION" \
  --initial-pressure-pct 5 \
  --min-pressure-pct 0 \
  --max-pressure-pct 85 \
  --pressure-step-pct 5 \
  --vm-workers 2 \
  --total-used-pct \
  --direct-cache-mode evict \
  --direct-cache-on-change \
  --pressure-spike-threshold-pct 92 \
  --pressure-calm-threshold-pct 72 \
  --top-k 100 \
  --ef-search 100

# G: workload sensitivity (lower ef_search)
run_profile "G_low_ef" "$STRESS_DURATION" \
  --initial-pressure-pct 5 \
  --min-pressure-pct 0 \
  --max-pressure-pct 85 \
  --pressure-step-pct 5 \
  --vm-workers 2 \
  --total-used-pct \
  --direct-cache-mode evict \
  --direct-cache-on-change \
  --pressure-spike-threshold-pct 92 \
  --pressure-calm-threshold-pct 72 \
  --top-k 10 \
  --ef-search 40

echo ""
echo "All A-G benchmark profiles completed. Outputs written to: ${OUT_DIR}"
