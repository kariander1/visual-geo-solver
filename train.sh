#!/bin/bash
set -e  # Exit on error

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  echo "Usage: $0 [NUM_GPUS] [CONFIG_NAME]"
  echo "  NUM_GPUS: Number of GPUs to use (1, 2, 4, or 8). Default: 2"
  echo "  CONFIG_NAME: Name of config file in configs/ directory (without .yaml). Default: config"
  echo ""
  echo "Examples:"
  echo "  $0                    # Use 2 GPUs with configs/config_curves.yaml"
  echo "  $0 4                  # Use 4 GPUs with configs/config_curves.yaml"
  echo "  $0 8 polygon          # Use 8 GPUs with configs/config_polygonalization_max.yaml"
  exit 0
fi

# === Parse command line arguments ===
NUM_GPUS=${1:-4}  # Default to 4 if not provided
CONFIG_NAME=${2:-config_curves}  # Default to 'config' if not provided

source .venv/bin/activate
echo "Current Python path: $(which python)"

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m scripts.train --config-name "$CONFIG_NAME" \
    hydra.run.dir="$RUN_DIR" \
