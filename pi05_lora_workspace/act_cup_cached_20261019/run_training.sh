#!/usr/bin/env bash
set -euo pipefail
cd /home/lp531/pi05_lora_workspace/act_cup_cached_20261019
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec /home/lp531/pi05_lora_workspace/.venv/bin/python -u train.py --train
