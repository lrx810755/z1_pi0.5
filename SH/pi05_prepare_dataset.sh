#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec /home/lp531/pi05_lora_workspace/.venv/bin/python /home/lp531/pi05_lora_workspace/pi05_current_round_20261111/pipeline.py prepare "$@"
