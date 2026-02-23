python benchmark_chroma_continuous_adaptive.py --path chroma_data --collection noise_test --query-source random --duration-seconds 100 --ramp-seconds 60 --pressure-end-pct 75 --top-k 10 --csv-out continuous_random_results.csv --max-latency-ms 500 --max-latency-count 1 --preload-cache --spike-threshold-ms 4.8 --ef-min 20 --ef-step-down 10 --csv-out continuous_random_adaptive_results.csv --consecutive-spikes-to-reduce 5 --ef-max 100 --ef-step-up 10

python benchmark_chroma_continuous.py --path chroma_data --collection noise_test --query-source random --duration-seconds 100 --ramp-seconds 60 --pressure-end-pct 75 --top-k 10 --csv-out continuous_random_results.csv --max-latency-ms 500 --max-latency-count 1 --preload-cache --csv-out continuous_random_results.csv 

python plot_continuous_results.py --csv continuous_random_results.csv --out continuous_random_results.png --latency-y-min 2 --latency-y-max 8 --window 200

python plot_continuous_results.py --csv continuous_random_adaptive_results.csv --out continuous_random_adaptive_results.png --latency-y-min 2 --latency-y-max 8 --window 200

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