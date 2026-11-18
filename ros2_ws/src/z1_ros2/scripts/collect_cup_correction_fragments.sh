#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=85
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json
exec /usr/bin/python3 /home/lp531/ros2_ws/src/z1_ros2/scripts/cup_correction_fragments_20261019/supervisor.py "$@"
