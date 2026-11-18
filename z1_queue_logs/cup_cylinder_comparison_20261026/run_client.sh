#!/usr/bin/env bash
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=94
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/lp531/z1_queue_logs/cup_cylinder_comparison_20261026/ros_overlay:/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge:${PYTHONPATH:-}"
exec /usr/bin/python3 /home/lp531/z1_queue_logs/cup_cylinder_comparison_20261026/run_cases.py "$@"
