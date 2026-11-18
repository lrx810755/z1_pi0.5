#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
set -u
exec /home/lp531/miniconda3/envs/unifolm-vla/bin/python \
  -m z1_isaac_bridge.smolvla_sim_rollout "$@"
