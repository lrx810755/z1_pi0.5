#!/usr/bin/env bash
set -eo pipefail
root=/home/lp531/robot/bottle_correction60_20261111
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=138
export Z1_COLLECTION_GRID_ROWS=4 Z1_COLLECTION_GRID_COLUMNS=5
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"
export Z1_ISAAC_GRASP_CONFIG="$root/grasp_physics.json"
if [[ "${1:-}" != --status && "${1:-}" != --dry-run ]]; then
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
  (( free_mib >= 18000 )) || { echo '可用显存不足18GB，未启动'; exit 1; }
  if pgrep -f '[/]run_z1_ros2_scene.py|[r]os2 launch .*z1_isaac' >/dev/null; then
    echo '检测到已有仿真，请先结束重复采集；本次未启动'; exit 1
  fi
fi
exec /usr/bin/python3 -u "$root/run.py" "$@"
