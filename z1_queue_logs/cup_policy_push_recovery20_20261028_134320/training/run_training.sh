#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/lp531/pi05_lora_workspace/lerobot/src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
exec /home/lp531/pi05_lora_workspace/.venv/bin/python -u workflow.py "$@"
