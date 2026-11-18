#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
log_dir=/home/lp531/z1_queue_logs/apple_column3_three_rounds_20261016
export PYTHONUNBUFFERED=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
setsid ros2 launch z1_isaac_bridge z1_isaac.launch.py \
 external_camera_layout:=dual rviz:=false task_gui:=true run_smolvla_policy:=false \
 freeze_target_keep_collisions:=true target_object:=apple \
 dataset_root:=/home/lp531/smolvla_data/z1_tabletop_raw_apple_preplanned_three_camera_20261016 \
 wrist_rgb_topic:=/left_wrist_camera/color/image_raw vla_wrist_rgb_topic:=none \
 recording_fps:=10.0 recording_sync_tolerance_s:=0.045 recording_sync_delay_s:=0.5 recording_action_age_s:=0.40 \
 >"$log_dir/scene.log" 2>&1 &
scene_pid=$!
trap 'kill -- -"$scene_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
/usr/bin/python3 "$log_dir/collect.py"
