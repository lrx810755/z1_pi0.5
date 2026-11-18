#!/usr/bin/env bash
set -euo pipefail
run_dir=${1:?usage: train_act_bottle_left_wrist_all_v1.sh OUTPUT_DIR}
dataset_root=/home/lp531/smolvla_data/z1_tabletop_lerobot_left_wrist_2cam_all_v1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
episodes=$(/usr/bin/python3 - "$dataset_root/meta/episodes.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1])]
ids=[row['episode_index'] for row in rows if row['tasks']==['Physically grasp the bottle, lift it clear, and place it in the box.']]
assert len(ids)==184 and sum(rows[i]['length'] for i in ids)==41252
print('['+','.join(map(str,ids))+']')
PY
)
exec /home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train \
  --dataset.repo_id=lp531/z1_smolvla_tabletop_left_wrist_2cam_all_v1 \
  "--dataset.root=$dataset_root" \
  "--dataset.episodes=$episodes" \
  --dataset.image_transforms.enable=false \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=10 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --batch_size=4 \
  --steps=30000 \
  --save_freq=3000 \
  --eval_freq=100000 \
  --log_freq=100 \
  --wandb.enable=false \
  "--output_dir=$run_dir"
