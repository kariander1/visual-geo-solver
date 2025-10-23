#!/bin/bash

data_ranges=("10_20_eval" "21_30_eval" "31_40_eval" "41_50_eval")

source .venv/bin/activate
mkdir -p evaluation_results

for data_range in "${data_ranges[@]}"; do
    echo "Evaluating ${data_range}"
    python scripts/evaluate_steiner.py \
        --checkpoint model/checkpoints_steiner/checkpoint_100_new.pth \
        --output-dir evaluation_results/steiner_${data_range} \
        --steiner-data-path data/steiner_data_eval/steiner_data_${data_range} \
        --limit 1000 \
        --best-of-n 10
done