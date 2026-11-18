#!/usr/bin/env bash
set -euo pipefail
RUN=/home/lp531/pi05_lora_workspace/act_cup_renderfix500_chunk30_batch64_20261028_003937
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/lp531/pi05_lora_workspace/lerobot/src
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
cd "$RUN"
exec /home/lp531/pi05_lora_workspace/.venv/bin/python -u workflow.py "$@"
