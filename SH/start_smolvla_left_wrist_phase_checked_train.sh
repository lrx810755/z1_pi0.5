#!/usr/bin/env bash
set -euo pipefail

source_root=/home/lp531/ros2_ws/src/z1_ros2
prepared=/home/lp531/smolvla_data/z1_tabletop_left_wrist_prepared_v1
base=/home/lp531/smolvla_data/z1_tabletop_lerobot_left_wrist_2cam_v1
state13=/home/lp531/smolvla_data/z1_tabletop_lerobot_left_wrist_state13_v1
repo_id=lp531/z1_smolvla_tabletop_left_wrist_state13_v1
log_dir=/home/lp531/z1_queue_logs/smolvla_left_wrist_v1
conda_bin=/home/lp531/miniconda3/bin/conda

[[ -f "$prepared/preprocessing_manifest.json" ]] || {
  echo "阶段审核和裁剪未完成：$prepared" >&2
  exit 1
}
mkdir -p "$log_dir" /home/lp531/smolvla_training

if [[ ! -f "$base/meta/info.json" ]]; then
  [[ ! -e "$base" ]] || { echo "LeRobot 转换目录不完整：$base" >&2; exit 1; }
  "$conda_bin" run --no-capture-output -n unifolm-vla python \
    "$source_root/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py" \
    --input "$prepared" --output "$base" \
    --repo-id lp531/z1_smolvla_tabletop_left_wrist_2cam_v1
fi

if [[ ! -f "$state13/meta/state_schema.json" ]]; then
  [[ ! -e "$state13" ]] || { echo "13 维状态目录不完整：$state13" >&2; exit 1; }
  "$conda_bin" run --no-capture-output -n unifolm-vla python \
    "$source_root/isaac_assets/z1/scripts/augment_z1_lerobot_state.py" \
    --source "$base" --raw "$prepared" --output "$state13" \
    --repo-id "$repo_id"
fi

"$conda_bin" run --no-capture-output -n unifolm-vla python - \
  "$prepared" "$state13" <<'PY'
import json
from pathlib import Path
import sys
import pyarrow.parquet as pq

raw, dataset = map(Path, sys.argv[1:])
episodes = sorted(raw.glob('episode_*'))
info = json.loads((dataset / 'meta/info.json').read_text())
assert info['total_episodes'] == len(episodes) == 518
assert info['total_frames'] == 132275
assert info['features']['observation.state']['shape'] == [13]
images = sorted(k for k in info['features'] if k.startswith('observation.images.'))
assert images == ['observation.images.global', 'observation.images.wrist'], images
phases = ('READY', 'GRASPING', 'PHYSICAL_TEST_LIFT', 'TRANSPORT', 'PLACING', 'SUCCESS')
for index, episode in enumerate(episodes):
    parquet = dataset / 'data/chunk-000' / f'episode_{index:06d}.parquet'
    states = pq.read_table(parquet, columns=['observation.state'])['observation.state'].to_pylist()
    with (episode / 'frames.jsonl').open() as stream:
        labels = [json.loads(line)['phase'] for line in stream]
    assert len(states) == len(labels), episode.name
    for frame_index, (state, label) in enumerate(zip(states, labels)):
        expected = [float(phase == label) for phase in phases]
        assert state[7:] == expected, (episode.name, frame_index, label)
print(f'validated {len(episodes)} episodes, {info["total_frames"]} frames, two cameras and every phase flag', flush=True)
PY

read -r frames episodes < <(python3 - "$state13/meta/info.json" <<'PY'
import json, sys
info=json.load(open(sys.argv[1]))
print(info['total_frames'], info['total_episodes'])
PY
)
batch_size=4
epochs=10
steps=$(( (frames * epochs + batch_size - 1) / batch_size ))
save_freq=$(( steps / 10 ))
stamp="$(date +%Y%m%d_%H%M%S)"
output="/home/lp531/smolvla_training/z1_left_wrist_phase_checked_${stamp}"
printf 'dataset=%s\nepisodes=%s\nframes=%s\nbatch_size=%s\nepochs=%s\nsteps=%s\noutput=%s\n' \
  "$state13" "$episodes" "$frames" "$batch_size" "$epochs" "$steps" "$output" \
  > "$log_dir/training_run.txt"

cd /home/lp531/lerobot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
/home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false \
  --policy.chunk_size=30 --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=false \
  --dataset.repo_id="$repo_id" --dataset.root="$state13" \
  --dataset.video_backend=pyav --output_dir="$output" \
  --job_name=z1_left_wrist_phase_checked \
  --batch_size="$batch_size" --num_workers=4 --steps="$steps" \
  --log_freq=200 --save_freq="$save_freq" --wandb.enable=false
