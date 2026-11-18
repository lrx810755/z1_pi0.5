#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=105
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/z1_queue_logs/cup_correction100_full_renderfix_20261027/grasp_physics_minus90.json
exec /usr/bin/python3 -u /home/lp531/z1_queue_logs/cup_correction100_full_renderfix_20261027/collector/run.py "$@"
