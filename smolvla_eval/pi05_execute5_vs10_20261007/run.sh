#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 Z1_PI05_INFERENCE_SEED=123
exec /home/lp531/miniconda3/envs/unifolm-vla/bin/python -u /home/lp531/smolvla_eval/pi05_execute5_vs10_20261007/run.py
