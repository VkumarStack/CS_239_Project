# Scripts ran on Adam's Machine:

## Vivek
- Benchmark memory pressure, no adaptive controller at ef-100, top-k 10
  - Spike: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 65 --pressure-profile spike --spike-peak-pct 33 --spike-rise-seconds 15 --spike-hold-seconds 20 --spike-fall-seconds 15 --spike-idle-seconds 20 --top-k 10 --read-segment-pct 100 --read-interval-ms 1 --ef-search 100 --csv-out outputs/non_adaptive_ef_comparison_ef_100_top_k_10_spike.csv`

  - Ramp: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 80 --pressure-profile ramp --ramp-seconds 60 --pressure-end-pct 60 --top-k 10 --read-segment-pct 100 --read-interval-ms 1 --ef-search 100 --csv-out outputs/non_adaptive_ef_comparison_ef_100_top_k_10_ramp.csv`

- Benchmark memory pressure, no adaptive controller at ef-20, top-k 10
  - Spike: `python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 65 --pressure-profile spike --spike-peak-pct 33 --spike-rise-seconds 15 --spike-hold-seconds 20 --spike-fall-seconds 15 --spike-idle-seconds 20 --top-k 10 --read-segment-pct 100 --read-interval-ms 1 --ef-search 20 --csv-out outputs/non_adaptive_ef_comparison_ef_20_top_k_10_spike.csv`

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