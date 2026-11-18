#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
raw_root="${BOTTLE_THREE_CAMERA_DATA_ROOT:-/home/lp531/smolvla_data/z1_tabletop_raw_bottle_three_camera_v1}"
if timeout 5 ros2 service list 2>/dev/null | grep -Fx /smolvla/recording/start >/dev/null; then
  echo '当前 ROS domain 已有采集场景，请先关闭该场景。' >&2
  exit 1
fi
exec ros2 launch z1_isaac_bridge z1_isaac.launch.py \
 external_camera_layout:=dual target_object:=beverage_bottle \
 dataset_root:="$raw_root" wrist_rgb_topic:=/left_wrist_camera/color/image_raw \
 vla_wrist_rgb_topic:=none recording_fps:=10.0 \
 recording_sync_tolerance_s:=0.045 recording_sync_delay_s:=0.5 \
 recording_action_age_s:=0.40 freeze_target_keep_collisions:=true \
 rviz:=false task_gui:=true run_smolvla_policy:=false "$@"
