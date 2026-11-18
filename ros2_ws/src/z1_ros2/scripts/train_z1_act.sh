#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {apple|peach|cup|beverage_bottle} [--dry-run]" >&2
  exit 2
fi

target=$1
case "$target" in
  apple|peach|cup|beverage_bottle) ;;
  *) echo "Unknown ACT target: $target" >&2; exit 2 ;;
esac
if [[ $# -eq 2 && $2 != --dry-run ]]; then
  echo "Unknown option: $2" >&2
  exit 2
fi

dataset_base=${ACT_DATASET_BASE:-/home/lp531/smolvla_data}
training_root=${ACT_TRAINING_ROOT:-/home/lp531/smolvla_training}
dataset_root="$dataset_base/z1_tabletop_lerobot_act_${target}_v1"
if [[ ! -s $dataset_root/meta/info.json || ! -s $dataset_root/meta/episodes.jsonl ]]; then
  echo "ACT dataset is missing or incomplete: $dataset_root" >&2
  exit 1
fi

run_name="act_${target}_$(date +%Y%m%d_%H%M%S)"
command=(
  /home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train
  "--dataset.repo_id=lp531/z1_act_${target}_v1"
  "--dataset.root=$dataset_root"
  --policy.type=act
  --policy.device=cuda
  --policy.chunk_size=30
  --policy.n_action_steps=10
  --batch_size="${ACT_BATCH_SIZE:-2}"
  --steps="${ACT_TRAIN_STEPS:-10000}"
  --save_freq=2000
  --wandb.enable=false
  "--output_dir=$training_root/$run_name"
)

if [[ ${2:-} == --dry-run ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$training_root"
"${command[@]}" 2>&1 | tee "$training_root/$run_name.log"
