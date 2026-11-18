#!/usr/bin/env bash
set -eo pipefail
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=94
export PYTHONPATH="/home/lp531/z1_queue_logs/cup_frustum_grasp_validation_20261026/ros_overlay:${PYTHONPATH:-}"
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/z1_queue_logs/cup_frustum_grasp_validation_20261026/grasp_physics.json
export PYTHONUNBUFFERED=1
exec ros2 launch /home/lp531/z1_queue_logs/cup_frustum_grasp_validation_20261026/collection.launch.py scene:=/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd isaac_script:=/home/lp531/z1_queue_logs/cup_frustum_grasp_validation_20261026/scene.py headless:=true rviz:=false task_gui:=true task_gui_port:=8810 target_object:=cup external_camera_layout:=dual run_smolvla_policy:=false freeze_target_keep_collisions:=true wrist_rgb_topic:=/left_wrist_camera/color/image_raw vla_wrist_rgb_topic:=none dataset_root:=/home/lp531/z1_queue_logs/cup_frustum_grasp_validation_20261026/inactive_recorder
