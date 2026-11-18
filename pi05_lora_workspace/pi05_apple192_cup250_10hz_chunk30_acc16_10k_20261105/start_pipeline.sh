#!/usr/bin/env bash
set -euo pipefail
cd /home/lp531/pi05_lora_workspace/pi05_apple192_cup250_10hz_chunk30_acc16_10k_20261105
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec /home/lp531/pi05_lora_workspace/.venv/bin/python -u pipeline.py
