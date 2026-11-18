#!/usr/bin/env bash
set -euo pipefail

dataset_root=/home/lp531/smolvla_data/z1_tabletop_lerobot_global_only_all_v1
run_dir=/home/lp531/smolvla_training/smolvla_bottle_global_only_batch8_20260929

if [[ -e $run_dir ]]; then
  echo "Training output already exists: $run_dir" >&2
  exit 1
fi

episodes=$(/usr/bin/python3 - "$dataset_root/meta/episodes.jsonl" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
ids = [row["episode_index"] for row in rows if row["tasks"] == ["Physically grasp the bottle, lift it clear, and place it in the box."]]
assert len(ids) == 184 and sum(rows[i]["length"] for i in ids) == 41252
print("[" + ",".join(map(str, ids)) + "]")
PY
)

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0

exec /home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train \
  --dataset.repo_id=lp531/z1_smolvla_tabletop_global_only_all_v1 \
  "--dataset.root=$dataset_root" \
  "--dataset.episodes=$episodes" \
  --dataset.image_transforms.enable=false \
  --dataset.video_backend=pyav \
  --policy.type=smolvla \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.use_amp=true \
  --policy.freeze_vision_encoder=true \
  --policy.chunk_size=30 \
  --policy.n_action_steps=10 \
  --batch_size=8 \
  --steps=30000 \
  --seed=1000 \
  --num_workers=4 \
  --save_freq=3000 \
  --eval_freq=100000 \
  --log_freq=20 \
  --wandb.enable=false \
  "--output_dir=$run_dir"
