#!/bin/bash

source .venv/bin/activate
mkdir -p evaluation_results

python scripts/eval_squares.py --checkpoint model/checkpoints_curves/checkpoint_520.pth --output-dir evaluation_results/squares_snapped