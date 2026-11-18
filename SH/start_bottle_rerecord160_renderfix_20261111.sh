#!/usr/bin/env bash
set -euo pipefail
root=/home/lp531/robot/bottle_rerecord160_renderfix_20261111
logs=/home/lp531/z1_queue_logs/bottle_rerecord160_renderfix_20261111
data=/home/lp531/smolvla_data/bottle_rerecord160_renderfix_20261111
preview=false
case "${1:-}" in
  --dry-run) preview=true ;;
  --status) /usr/bin/python3 - "$data" <<'PY'
from pathlib import Path
import sys
print('Saved episodes:',len(list(Path(sys.argv[1]).glob('episode_*/episode_summary.json'))))
PY
    exit ;;
  '') ;;
  *) echo '支持无参数启动、--dry-run、--status';exit 2 ;;
esac
source /home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh
export ROS_DOMAIN_ID=140
export Z1_COLLECTION_GRID_ROWS=4 Z1_COLLECTION_GRID_COLUMNS=5
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export Z1_ISAAC_GRASP_CONFIG="$root/grasp_physics.json"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
if ! $preview;then
  mkdir -p "$logs"
  exec 9>"$logs/.collection.lock"
  flock -n 9 || { echo '本轮160条采集已在运行';exit 1; }
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
  (( free_mib >= 18000 )) || { echo '可用显存不足18GB，未启动';exit 1; }
  if pgrep -f '[/]run_z1_ros2_scene.py|[r]os2 launch .*z1_isaac' >/dev/null;then
    echo '检测到已有仿真任务，本轮未启动';exit 1
  fi
fi
args=()
if $preview;then args+=(--dry-run);fi
exec bash "$root/start_collection.sh" "${args[@]}" 8 20 "$data" "$logs" /left_wrist_camera/color/image_raw none beverage_bottle
