#!/usr/bin/env bash
set -euo pipefail

reviewed_raw="/home/lp531/smolvla_data/z1_tabletop_reviewed_4x100_v1"
prepared_raw="/home/lp531/smolvla_data/z1_tabletop_reviewed_4x100_trimmed_2cam_v1"
lerobot_base="/home/lp531/smolvla_data/z1_tabletop_lerobot_4x100_2cam_trimmed_v1"
lerobot_state13="/home/lp531/smolvla_data/z1_tabletop_lerobot_4x100_state13_2cam_trimmed_v1"
repo_id="lp531/z1_smolvla_tabletop_4x100_state13_2cam_trimmed"
prepare_script="/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/prepare_smolvla_dataset.py"
export_script="/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py"
augment_script="/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/augment_z1_lerobot_state.py"
conda_bin="/home/lp531/miniconda3/bin/conda"

if [[ ! -f "${prepared_raw}/preprocessing_manifest.json" ]]; then
  [[ ! -e "${prepared_raw}" ]] || {
    echo "Incomplete prepared dataset exists: ${prepared_raw}" >&2
    exit 1
  }
  /usr/bin/python3 "${prepare_script}" \
    --source "${reviewed_raw}" \
    --output "${prepared_raw}" \
    --static-threshold 0.002
fi

if [[ ! -f "${lerobot_base}/meta/info.json" ]]; then
  [[ ! -e "${lerobot_base}" ]] || {
    echo "Incomplete LeRobot base dataset exists: ${lerobot_base}" >&2
    exit 1
  }
  "${conda_bin}" run --no-capture-output -n unifolm-vla python \
    "${export_script}" \
    --input "${prepared_raw}" \
    --output "${lerobot_base}" \
    --repo-id lp531/z1_smolvla_tabletop_4x100_2cam_trimmed
fi

if [[ ! -f "${lerobot_state13}/meta/state_schema.json" ]]; then
  [[ ! -e "${lerobot_state13}" ]] || {
    echo "Incomplete state13 dataset exists: ${lerobot_state13}" >&2
    exit 1
  }
  "${conda_bin}" run --no-capture-output -n unifolm-vla python \
    "${augment_script}" \
    --source "${lerobot_base}" \
    --raw "${prepared_raw}" \
    --output "${lerobot_state13}" \
    --repo-id "${repo_id}"
fi

"${conda_bin}" run --no-capture-output -n unifolm-vla python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
d = LeRobotDataset('${repo_id}', root='${lerobot_state13}')
assert d.num_episodes == 400, d.num_episodes
assert d.meta.features['observation.state']['shape'] == (13,)
images = sorted(k for k in d.meta.features if k.startswith('observation.images.'))
assert images == ['observation.images.global', 'observation.images.wrist'], images
print(f'dataset ready: episodes={d.num_episodes} frames={d.num_frames} cameras={images}')
"

training_stamp="$(date +%Y%m%d_%H%M%S)"
training_output="/home/lp531/smolvla_training/z1_4x100_state13_2cam_trimmed_chunk30_${training_stamp}"
training_log="/home/lp531/z1_queue_logs/smolvla_state13_2cam_trimmed_chunk30_train.log"

mkdir -p /home/lp531/smolvla_training /home/lp531/z1_queue_logs
cd /home/lp531/lerobot

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=10 \
  --dataset.repo_id="${repo_id}" \
  --dataset.root="${lerobot_state13}" \
  --dataset.video_backend=pyav \
  --output_dir="${training_output}" \
  --job_name=z1_4x100_state13_2cam_trimmed_chunk30 \
  --batch_size=4 \
  --num_workers=4 \
  --steps=100000 \
  --log_freq=200 \
  --save_freq=10000 \
  --wandb.enable=false \
  2>&1 | tee "${training_log}"
