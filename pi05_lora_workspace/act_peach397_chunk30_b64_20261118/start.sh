#!/usr/bin/env bash
set -euo pipefail
cd /home/lp531/pi05_lora_workspace/act_peach397_chunk30_b64_20261118
exec env -u PYTHONPATH -u PYTHONHOME OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /home/lp531/pi05_lora_workspace/.venv/bin/python -u workflow.py
