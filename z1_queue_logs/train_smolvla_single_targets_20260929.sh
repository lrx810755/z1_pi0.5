#!/usr/bin/env bash
set -Eeuo pipefail

dataset_root=/home/lp531/smolvla_data/z1_tabletop_lerobot_left_wrist_2cam_all_v1
train_bin=/home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train
run_root=/home/lp531/smolvla_training/single_target_20260929
log_root=/home/lp531/z1_queue_logs/single_target_20260929
dry_run=${1:-}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0

for target in cup apple peach; do
  episodes=$(/usr/bin/python3 - "$dataset_root/meta/episodes.jsonl" "$target" <<'PY'
import json
import sys

path, target = sys.argv[1:]
task = f"Physically grasp the {target}, lift it clear, and place it in the box."
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
ids = [row["episode_index"] for row in rows if row["tasks"] == [task]]
if not ids or len(ids) != len(set(ids)):
    raise SystemExit(f"invalid {target} episode selection: {len(ids)} rows")
print("[" + ",".join(map(str, ids)) + "]")
PY
  )
  run_dir="$run_root/$target"
  args=(
    --dataset.repo_id=lp531/z1_smolvla_tabletop_left_wrist_2cam_all_v1
    "--dataset.root=$dataset_root"
    "--dataset.episodes=$episodes"
    --dataset.image_transforms.enable=false
    --dataset.video_backend=pyav
    --policy.type=smolvla
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.use_amp=true
    --policy.freeze_vision_encoder=true
    --policy.chunk_size=30
    --policy.n_action_steps=10
    --batch_size=8
    --steps=30000
    --seed=1000
    --num_workers=4
    --save_freq=3000
    --eval_freq=100000
    --log_freq=20
    --wandb.enable=false
    "--output_dir=$run_dir"
  )
  if [[ $dry_run == --dry-run ]]; then
    count=$(tr -cd , <<< "$episodes" | wc -c)
    printf '%s episodes=%s output=%s\n' "$target" "$((count + 1))" "$run_dir"
    printf '%q ' "$train_bin" "${args[@]}"
    printf '\n'
    continue
  fi
  mkdir -p "$log_root"
  if [[ -e $run_dir ]]; then
    printf '%s %s: output already exists; refusing to overwrite\n' "$(date -Is)" "$target" | tee -a "$log_root/queue.log"
    continue
  fi
  printf '%s %s: starting SmolVLA training\n' "$(date -Is)" "$target" | tee -a "$log_root/queue.log"
  if "$train_bin" "${args[@]}" > "$log_root/$target.log" 2>&1; then
    printf '%s %s: completed\n' "$(date -Is)" "$target" | tee -a "$log_root/queue.log"
  else
    printf '%s %s: failed; see %s\n' "$(date -Is)" "$target" "$log_root/$target.log" | tee -a "$log_root/queue.log"
  fi
done
