#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
WORK=/home/lp531/smolvla_eval/pi05_sampling_ab_comparison_20261003
cd "$WORK"
for ARM in baseline uniform weighted; do
    /home/lp531/miniconda3/envs/unifolm-vla/bin/python "$WORK/evaluate.py" --arm "$ARM" >> "$WORK/${ARM}.log" 2>&1
done
