#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=121
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/z1_queue_logs/cup_policy_push_recovery20_20261028_134320/grasp_physics_minus90.json
export Z1_PI05_INFERENCE_SEED=20261101
exec /usr/bin/python3 "$(dirname "$0")/supervise.py"
