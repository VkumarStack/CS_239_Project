# Scripts ran on Adam's Machine:

## Vivek
- Benchmark memory pressure, no adaptive controller at ef-100, top-k 10
  - Spike: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 65 --pressure-profile spike --spike-peak-pct 33 --spike-rise-seconds 15 --spike-hold-seconds 20 --spike-fall-seconds 15 --spike-idle-seconds 20 --top-k 10 --read-segment-pct 100 --read-interval-ms 1 --ef-search 100 --csv-out outputs/non_adaptive_ef_comparison_ef_100_top_k_10_spike.csv`

  - Ramp: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 80 --pressure-profile ramp --ramp-seconds 60 --pressure-end-pct 60 --top-k 10 --read-segment-pct 100 --read-interval-ms 1 --ef-search 100 --csv-out outputs/non_adaptive_ef_comparison_ef_100_top_k_10_ramp.csv`

- Benchmark memory pressure, no adaptive controller at ef-20, top-k 10
  - Spike: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 65 --pressure-profile spike --spike-peak-pct 32 --spike-rise-seconds 15 --spike-hold-seconds 20 --spike-fall-seconds 15 --spike-idle-seconds 20 --top-k 10 --read-segment-pct 100 --read-interval-ms 1 --ef-search 20 --csv-out outputs/non_adaptive_ef_comparison_ef_20_top_k_10_spike.csv`

  - Ramp: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 80 --pressure-profile ramp --ramp-seconds 60 --pressure-end-pct 60 --top-k 10 --read-segment-pct 100 --read-interval-ms 1 --ef-search 20 --csv-out outputs/non_adaptive_ef_comparison_ef_20_top_k_10_ramp.csv`

- Benchmark memory pressure, no adaptive controller at ef-100, top-k 100
  - Spike: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 65 --pressure-profile spike --spike-peak-pct 33 --spike-rise-seconds 15 --spike-hold-seconds 20 --spike-fall-seconds 15 --spike-idle-seconds 20 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --ef-search 100 --csv-out outputs/non_adaptive_ef_comparison_ef_100_top_k_100_spike.csv`

  - Ramp: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 80 --pressure-profile ramp --ramp-seconds 60 --pressure-end-pct 60 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --ef-search 100 --csv-out outputs/non_adaptive_ef_comparison_ef_100_top_k_100_ramp.csv`

- Benchmark memory pressure, no adaptive controller at ef-20, top-k 100
  - Spike: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 65 --pressure-profile spike --spike-peak-pct 33 --spike-rise-seconds 15 --spike-hold-seconds 20 --spike-fall-seconds 15 --spike-idle-seconds 20 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --ef-search 20 --csv-out outputs/non_adaptive_ef_comparison_ef_20_top_k_100_spike.csv`

  - Ramp: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 80 --pressure-profile ramp  --ramp-seconds 60 --pressure-end-pct 60 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --ef-search 20 --csv-out outputs/non_adaptive_ef_comparison_ef_20_top_k_100_ramp.csv`

- Benchmark memory pressure, adaptive controller, top-k 10
  - Spike: `python benchmark_chroma_continuous_adaptive.py --path chroma_data --collection noise_test --query-source random --duration-seconds 65 --pressure-profile spike --spike-peak-pct 32 --spike-rise-seconds 15 --spike-hold-seconds 20 --spike-fall-seconds 15 --spike-idle-seconds 20 --top-k 10 --spike-threshold-ms 3.5 --ef-min 20 --ef-step-down 10 --consecutive-spikes-to-reduce 10 --ef-max 100 --ef-step-up 10 --consecutive-calm-to-increase 3 --read-segment-pct 33 --read-interval-ms 10 --csv-out outputs/adaptive_ef_comparison_top_10_spike.csv`
  - Ramp: `python benchmark_chroma_continuous_adaptive.py --path chroma_data --collection noise_test --query-source random --duration-seconds 80 --pressure-profile ramp  --ramp-seconds 60 --pressure-end-pct 60 --top-k 10 --spike-threshold-ms 3.5 --ef-min 20 --ef-step-down 10 --consecutive-spikes-to-reduce 10 --ef-max 100 --ef-step-up 10 --consecutive-calm-to-increase 3 --read-segment-pct 33 --read-interval-ms 10 --csv-out outputs/adaptive_ef_comparison_top_10_ramp.csv`

- Benchmark memory pressure, adaptive controller top-k 100
  - Spike: `python benchmark_chroma_continuous_adaptive.py --path chroma_data --collection noise_test --query-source random --duration-seconds 65 --pressure-profile spike --spike-peak-pct 32 --spike-rise-seconds 15 --spike-hold-seconds 20 --spike-fall-seconds 15 --spike-idle-seconds 20 --top-k 100 --spike-threshold-ms 3.5 --ef-min 20 --ef-step-down 10 --consecutive-spikes-to-reduce 10 --ef-max 100 --ef-step-up 10 --consecutive-calm-to-increase 3 --read-segment-pct 33 --read-interval-ms 10 --csv-out outputs/adaptive_ef_comparison_top_100_spike.csv`
  - Ramp: `python benchmark_chroma_continuous_adaptive.py --path chroma_data --collection noise_test --query-source random --duration-seconds 80 --pressure-profile ramp  --ramp-seconds 60 --pressure-end-pct 60 --top-k 100 --spike-threshold-ms 3.5 --ef-min 20 --ef-step-down 10 --consecutive-spikes-to-reduce 10 --ef-max 100 --ef-step-up 10 --consecutive-calm-to-increase 3 --read-segment-pct 33 --read-interval-ms 10 --csv-out outputs/adaptive_ef_comparison_top_100_ramp.csv`

- FAISS Float32 Baseline top-k-500
  - Ramp: `python benchmark_faiss_baseline.py --index-type float32 --duration-seconds 60 --pressure-profile ramp --ramp-seconds 40 --pressure-end-pct 95 --top-k 500 --csv-out outputs/faiss_baseline_top_500_ramp.csv`

  - FAISS int8 Baseline top-k-500
  - Ramp: `python benchmark_faiss_baseline.py --index-type int8 --duration-seconds 60 --pressure-profile ramp --ramp-seconds 40 --pressure-end-pct 95 --top-k 500 --csv-out outputs/faiss_baseline_int8_top_500_ramp.csv`


# Works on Marco's machine, had to set up chroma data outside of mounted dir for true page eviction behavior

python benchmark_chroma_cache_pressure.py \
  --path /home/ubuntu/chroma_data \
  --mem-steps 0,40,60,75,85,90 \
  --vm-workers 2 \
  --queries-per-step 200 \
  --query-mode fixed \
  --preload-cache \
  --track-vmtouch \
  --csv-out /home/ubuntu/results_eviction.csv \
  --timeline-out /home/ubuntu/results_eviction_timeline.csv

## Continuous adaptive cache-pressure benchmark
python3 benchmark_chroma_cache_continuous_adaptive.py \
  --path /home/ubuntu/chroma_data \
  --duration-seconds 120 \
  --initial-pressure-pct 0 \
  --max-pressure-pct 95 \
  --pressure-step-pct 5 \
  --vm-workers 2 \
  --eval-window-queries 200 \
  --csv-out continuous_cache_adaptive.csv

# runs on VM
python3 benchmark_chroma_cache_continuous_adaptive.py \
  --path chroma_data \
  --duration-seconds 120 \
  --initial-pressure-pct 0 \
  --max-pressure-pct 95 \
  --pressure-step-pct 5 \
  --vm-workers 2 \
  --eval-window-queries 200 \
  --csv-out outputs/continuous_cache_adaptive.csv

  python benchmark_v2_chroma_cache_continuous_adaptive.py   --path chroma_data   --duration-seconds 180   --query-pool-size 500   --query-interval-ms 0   --disable-stress   --direct-cache-mode evict   --direct-cache-on-start   --direct-cache-on-change   --direct-cache-every-n-queries 50   --spike-threshold-ms 200   --calm-threshold-ms 80   --pressure-spike-threshold-pct 95   --pressure-calm-threshold-pct 75   --csv-out outputs/continuous_cache_adaptive_v2.csv   --timeline-out outputs/continuous_cache_adaptive_v2_timeline.csv

  python benchmark_v2_chroma_cache_pressure.py   --path chroma_data  --mem-steps 0,0,0,0   --queries-per-step 200   --query-mode fixed   --direct-cache-mode evict   --direct-cache-before-each-step   --direct-cache-min-size-bytes 4096   --track-vmtouch   --csv-out /home/ubuntu/results_pagecache_only.csv   --timeline-out /home/ubuntu/results_pagecache_only_timeline.csv

## v2 adaptive cache workload plot (latency + controller + QPS)
python plot_continuous_cache_adaptive_v2.py \
  --csv outputs/continuous_cache_adaptive_v2.csv \
  --out outputs/continuous_cache_adaptive_v2.png \
  --latency-window 150 \
  --qps-window 120
