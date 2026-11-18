#!/bin/bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
exec /home/lp531/miniconda3/envs/unifolm-vla/bin/python /home/lp531/smolvla_eval/pi05_four_step_validation_20261003/evaluate_corrected.py
