#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -euo pipefail
unset PYTHONHOME
export ROS_DOMAIN_ID=168
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/robot/pi05_four_objects_eval_20261117/grasp_physics.json
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge:${PYTHONPATH:-}"
if [[ "${1:-}" == model ]]; then
  shift
  unset PYTHONNOUSERSITE
fi
exec "$@"
