# Scripts ran on Adam's Machine:

## Adaptive ef Controller on Ramp
python benchmark_chroma_continuous_adaptive.py --path chroma_data --collection noise_test --query-source random --duration-seconds 100 --ramp-seconds 80 --pressure-end-pct 95 --top-k 100 --spike-threshold-ms 3.5 --ef-min 20 --ef-step-down 10 --consecutive-spikes-to-reduce 10 --ef-max 100 --ef-step-up 10 --consecutive-calm-to-increase 3 --read-segment-pct 33 --read-interval-ms 10 --csv-out outputs/3_2_2026_experiment1_adaptive.csv

python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random  --duration-seconds 80 --ramp-seconds 40 --pressure-end-pct 95 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --csv-out outputs/3_2_2026_experiment1.csv

## Comparison of ef_search = 100 vs ef_search = 20 
python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random  --duration-seconds 80 --ramp-seconds 40 --pressure-end-pct 95 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --ef-search 100 --csv-out outputs/3_2_2026_experiment2_ef100.csv

python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random  --duration-seconds 80 --ramp-seconds 40 --pressure-end-pct 95 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --ef-search 20 --csv-out outputs/3_2_2026_experiment2_ef20.csv

## Periodic Ramp
python benchmark_chroma_continuous_adaptive.py --path chroma_data --collection noise_test --query-source random --duration-seconds 120 --pressure-profile spike --spike-baseline-pct 30 --spike-peak-pct 95 --spike-rise-seconds 1 --spike-hold-seconds 3 --spike-fall-seconds 1 --spike-idle-seconds 20 --top-k 100 --spike-threshold-ms 3.5 --ef-min 20 --ef-step-down 10 --consecutive-spikes-to-reduce 10 --ef-max 100 --ef-step-up 10 --consecutive-calm-to-increase 3 --read-segment-pct 100 --read-interval-ms 1 --csv-out outputs/3_2_2026_experiment3_adaptive.csv

python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 120 --pressure-profile spike --spike-baseline-pct 30 --spike-peak-pct 95 --spike-rise-seconds 1 --spike-hold-seconds 3 --spike-fall-seconds 1 --spike-idle-seconds 20 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --csv-out outputs/3_2_2026_experiment3.csv

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
python benchmark_chroma_cache_continuous_adaptive.py \
  --path /home/ubuntu/chroma_data \
  --duration-seconds 150 \
  --query-pool-size 500 \
  --vm-workers 2 \
  --initial-pressure-pct 40 \
  --min-pressure-pct 0 \
  --max-pressure-pct 80 \
  --pressure-step-pct 5 \
  --eval-window-queries 200 \
  --spike-threshold-ms 150 \
  --calm-threshold-ms 50 \
  --csv-out /home/ubuntu/CS_239_Project/continuous_cache_adaptive_2.csv

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