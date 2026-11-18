#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=131
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/robot/apple_scale110_correction60_20261106/grasp_physics.json
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
exec /usr/bin/python3 /home/lp531/robot/apple_scale110_correction60_20261106/supervisor.py --output /home/lp531/smolvla_data/apple_scale110_correction60_20261106 --port 8834 --until-complete "$@"
