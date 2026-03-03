# Scripts ran on Adam's Machine:

## Adaptive ef Controller on Ramp
python benchmark_chroma_continuous_adaptive.py --path chroma_data --collection noise_test --query-source random --duration-seconds 100 --ramp-seconds 80 --pressure-end-pct 95 --top-k 10 --spike-threshold-ms 3.5 --ef-min 20 --ef-step-down 10 --consecutive-spikes-to-reduce 10 --ef-max 100 --ef-step-up 10 --consecutive-calm-to-increase 3 --read-segment-pct 33 --read-interval-ms 10 --csv-out outputs/3_2_2026_experiment1_adaptive.csv

python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random  --duration-seconds 80 --ramp-seconds 40 --pressure-end-pct 95 --top-k 100 --read-segment-pct 100 --read-interval-ms 1 --csv-out outputs/3_2_2026_experiment1.csv

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