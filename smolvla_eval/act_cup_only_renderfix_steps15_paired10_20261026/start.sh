#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export PYTHONPATH="/home/lp531/smolvla_eval/act_cup_only_renderfix_steps15_paired10_20261026/ros_overlay:${PYTHONPATH:-}"
export PYTHONPATH="/home/lp531/smolvla_eval/act_cup_only_renderfix_steps15_paired10_20261026/ros_overlay:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=98
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
exec /usr/bin/python3 "$(dirname "$0")/supervise.py"
