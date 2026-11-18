#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
exec ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  isaac_script:=/home/lp531/smolvla_eval/grasp_parameter_comparison_20261008/scene.py \
  external_camera_layout:=dual target_object:=beverage_bottle \
  rviz:=false task_gui:=true run_smolvla_policy:=false "$@"
