#!/usr/bin/env bash
set -euo pipefail

object="${1:-apple}"
episodes="${2:-1}"
output="${3:-/home/lp531/smolvla_eval/collected_state13_chunk30}"
training_root="/home/lp531/smolvla_training/z1_collected_state13_chunk30_20260925_113739"
checkpoint_arg="${4:-367333}"
if [[ "$checkpoint_arg" == /* ]]; then
  checkpoint="$checkpoint_arg"
else
  checkpoint="$training_root/checkpoints/$checkpoint_arg/pretrained_model"
fi
[[ -d "$checkpoint" ]] || {
  echo "找不到 checkpoint: $checkpoint（最终模型需等待训练完成）" >&2
  exit 1
}

exec /home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_sim_eval.sh \
  --checkpoint "${checkpoint}" \
  --training-dataset-root /home/lp531/smolvla_data/z1_tabletop_raw_balanced_v2 \
  --objects "${object}" \
  --episodes-per-object "${episodes}" \
  --output "${output}"
