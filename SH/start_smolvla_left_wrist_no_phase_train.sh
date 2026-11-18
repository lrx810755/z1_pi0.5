#!/usr/bin/env bash
set -euo pipefail

export_pid="${1:-}"
log_dir=/home/lp531/z1_queue_logs/smolvla_left_wrist_v1
conversion_log="${CONVERSION_LOG:-$log_dir/train_pipeline.log}"
dataset="${DATASET_ROOT:-/home/lp531/smolvla_data/z1_tabletop_lerobot_left_wrist_2cam_v1}"
repo_id="${REPO_ID:-lp531/z1_smolvla_tabletop_left_wrist_2cam_v1}"
expected_episodes="${EXPECTED_EPISODES:-518}"
expected_frames="${EXPECTED_FRAMES:-132275}"
output_prefix="${OUTPUT_PREFIX:-z1_left_wrist_no_phase}"
conda_bin=/home/lp531/miniconda3/bin/conda

while ! grep -Fq "LeRobot dataset ready: $dataset" "$conversion_log"; do
  if [[ -n "$export_pid" ]] && ! kill -0 "$export_pid" 2>/dev/null; then
    echo "LeRobot 转换进程已退出，但没有完成标记；查看 $conversion_log" >&2
    exit 1
  fi
  sleep 30
done

"$conda_bin" run --no-capture-output -n unifolm-vla python - \
  "$dataset" "$repo_id" "$expected_episodes" "$expected_frames" <<'PY'
import json
from pathlib import Path
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(sys.argv[1])
info = json.loads((root / 'meta/info.json').read_text())
expected_episodes, expected_frames = map(int, sys.argv[3:])
assert info['total_episodes'] == expected_episodes, info['total_episodes']
assert info['total_frames'] == expected_frames, info['total_frames']
assert info['features']['observation.state']['shape'] == [7]
images = sorted(k for k in info['features'] if k.startswith('observation.images.'))
assert images == ['observation.images.global', 'observation.images.wrist'], images
dataset = LeRobotDataset(sys.argv[2], root=root)
assert dataset.num_episodes == expected_episodes and dataset.num_frames == expected_frames
print(f'validated: {expected_episodes} episodes, {expected_frames} frames, 7 joint states, two cameras', flush=True)
PY

read -r frames episodes < <(python3 - "$dataset/meta/info.json" <<'PY'
import json, sys
info=json.load(open(sys.argv[1]))
print(info['total_frames'], info['total_episodes'])
PY
)
batch_size="${BATCH_SIZE:-4}"
epochs=10
steps=$(( (frames * epochs + batch_size - 1) / batch_size ))
save_freq=$(( steps / 10 ))
stamp="$(date +%Y%m%d_%H%M%S)"
output="/home/lp531/smolvla_training/${output_prefix}_${stamp}"
printf 'dataset=%s\nepisodes=%s\nframes=%s\nstate_dims=7\nbatch_size=%s\nepochs=%s\nsteps=%s\noutput=%s\n' \
  "$dataset" "$episodes" "$frames" "$batch_size" "$epochs" "$steps" "$output" \
  > "$log_dir/training_run_${output_prefix}.txt"

cd /home/lp531/lerobot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
/home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=false \
  --dataset.repo_id="$repo_id" --dataset.root="$dataset" \
  --dataset.video_backend=pyav --output_dir="$output" \
  --job_name="$output_prefix" \
  --batch_size="$batch_size" --num_workers=4 --steps="$steps" \
  --log_freq=200 --save_freq="$save_freq" --wandb.enable=false
