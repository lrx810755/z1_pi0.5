#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
cd /home/lp531/pi05_lora_workspace/diffusion_cup_gentle_cached_20261020
exec /home/lp531/pi05_lora_workspace/.venv/bin/python supervise.py
