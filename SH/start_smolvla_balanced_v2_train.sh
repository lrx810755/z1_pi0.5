#!/usr/bin/env bash
set -euo pipefail

raw="/home/lp531/smolvla_data/z1_tabletop_raw_balanced_v2"
prepared="/home/lp531/smolvla_data/z1_tabletop_collected_prepared_v2"
lerobot_base="/home/lp531/smolvla_data/z1_tabletop_lerobot_collected_2cam_v2"
lerobot_state13="/home/lp531/smolvla_data/z1_tabletop_lerobot_collected_state13_v2"
repo_id="lp531/z1_smolvla_tabletop_collected_state13_v2"
source_root="/home/lp531/ros2_ws/src/z1_ros2"
package_python="$source_root/z1_isaac_bridge"
prepare_script="$package_python/z1_isaac_bridge/prepare_smolvla_dataset.py"
export_script="$source_root/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py"
augment_script="$source_root/isaac_assets/z1/scripts/augment_z1_lerobot_state.py"
conda_bin="/home/lp531/miniconda3/bin/conda"
batch_size=4
epochs=10

[[ -f "$raw/dataset_info.json" ]] || {
  echo "找不到新原始数据：$raw；请先运行 /home/lp531/start_smolvla_balanced_v2_collection.sh" >&2
  exit 1
}

if [[ ! -f "$prepared/preprocessing_manifest.json" ]]; then
  [[ ! -e "$prepared" ]] || { echo "不完整目录已存在：$prepared" >&2; exit 1; }
  /usr/bin/python3 "$prepare_script" \
    --source "$raw" --output "$prepared" \
    --static-threshold 0.002 --leading-context-frames 3 \
    --max-action-step-rad 0.10 --terminal-hold-frames 10 \
    --max-identical-run-frames 3
fi

if [[ ! -f "$lerobot_base/meta/info.json" ]]; then
  [[ ! -e "$lerobot_base" ]] || { echo "不完整目录已存在：$lerobot_base" >&2; exit 1; }
  "$conda_bin" run --no-capture-output -n unifolm-vla python "$export_script" \
    --input "$prepared" --output "$lerobot_base" \
    --repo-id lp531/z1_smolvla_tabletop_collected_2cam_v2
fi

if [[ ! -f "$lerobot_state13/meta/state_schema.json" ]]; then
  [[ ! -e "$lerobot_state13" ]] || { echo "不完整目录已存在：$lerobot_state13" >&2; exit 1; }
  "$conda_bin" run --no-capture-output -n unifolm-vla python "$augment_script" \
    --source "$lerobot_base" --raw "$prepared" --output "$lerobot_state13" \
    --repo-id "$repo_id"
fi

read -r frame_count episode_count < <(/usr/bin/python3 - <<PY
import json
info=json.load(open('$lerobot_state13/meta/info.json'))
print(info['total_frames'], info['total_episodes'])
PY
)
(( episode_count > 0 && frame_count > 0 )) || { echo "导出数据为空" >&2; exit 1; }
steps=$(( (frame_count * epochs + batch_size - 1) / batch_size ))
save_freq=$(( steps / 10 ))
(( save_freq > 0 )) || save_freq=10000

stamp="$(date +%Y%m%d_%H%M%S)"
output="/home/lp531/smolvla_training/z1_collected_state13_chunk30_${stamp}"
log="/home/lp531/z1_queue_logs/smolvla_balanced_v2_train.log"
mkdir -p /home/lp531/smolvla_training /home/lp531/z1_queue_logs
cd /home/lp531/lerobot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "dataset: episodes=$episode_count frames=$frame_count；训练 $epochs epochs = $steps steps"
/home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=false \
  --dataset.repo_id="$repo_id" --dataset.root="$lerobot_state13" \
  --dataset.video_backend=pyav --output_dir="$output" \
  --job_name=z1_collected_state13_chunk30 \
  --batch_size="$batch_size" --num_workers=4 --steps="$steps" \
  --log_freq=200 --save_freq="$save_freq" --wandb.enable=false \
  2>&1 | tee "$log"
