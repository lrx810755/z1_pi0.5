#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
exec /usr/bin/python3 /home/lp531/ros2_ws/install/z1_isaac_bridge/lib/z1_isaac_bridge/grasp_evaluator --ros-args --params-file /tmp/launch_params_g2fsmosz --params-file /tmp/launch_params_649p0n1k
