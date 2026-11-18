#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=119
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export PYTHONUNBUFFERED=1
exec /usr/bin/python3 -u /home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031/pipeline.py
