#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /home/lp531/pi05_lora_workspace/.venv/bin/python /home/lp531/pi05_lora_workspace/pi05_current_round_20261111/pipeline.py status "$@"
