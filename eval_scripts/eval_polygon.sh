#!/bin/bash

mkdir -p evaluation_results

source .venv/bin/activate
echo "Current Python path: $(which python)"
python scripts/evaluate_polygonization.py --checkpoint model/checkpoints_max_polygon/checkpoint_100.pth --output-dir evaluation_results/polygon_7_12 --polygon-data-path data/polygon_data_7_12_max --limit 1000 --best-of-n 10
python scripts/evaluate_polygonization.py --checkpoint model/checkpoints_max_polygon/checkpoint_100.pth --output-dir evaluation_results/polygon_13_15 --polygon-data-path data/polygon_data_eval_13_15  --limit 1000 --best-of-n 10 --use-whole-dataset