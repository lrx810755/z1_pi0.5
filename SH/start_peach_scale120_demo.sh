#!/usr/bin/env bash
set -euo pipefail
root=/home/lp531/robot/peach_scale120_demo_20261106
logs=/home/lp531/z1_queue_logs/peach_scale120_demo_20261106
data=/home/lp531/smolvla_data/peach_scale120_demo_20261106
export ROS_DOMAIN_ID=132
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export Z1_ISAAC_GRASP_CONFIG="$root/grasp_physics.json"
export Z1_COLLECTION_GRID_ROWS=5 Z1_COLLECTION_GRID_COLUMNS=4
quota="${Z1_PEACH_QUOTA_PER_CELL:-20}"
[[ "$quota" =~ ^[1-9][0-9]*$ ]] || { echo '每格配额必须是正整数';exit 2; }
case "${1:-}" in
 --status) /usr/bin/python3 - "$data" <<'PY'
from pathlib import Path
import sys
print('Saved summaries:',len(list(Path(sys.argv[1]).glob('episode_*/episode_summary.json'))))
PY
 exit;;
 ''|--dry-run) ;;
 *) echo '支持 --dry-run、--status；Z1_PEACH_QUOTA_PER_CELL 可指定每格配额';exit 2;;
esac
mkdir -p "$logs"
cp "$root/collection.launch.py" "$logs/collection.launch.py"
if [[ "${1:-}" != --dry-run ]];then
 exec 9>"$logs/.collection.lock"
 flock -n 9 || { echo '此桃子入口已运行';exit 1; }
 free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
 (( free_mib >= 18000 )) || { echo '显存不足或已有仿真/训练，请先关闭重复任务';exit 1; }
fi
args=()
if [[ "${1:-}" == --dry-run ]];then args+=(--dry-run);fi
exec bash "$root/start_collection.sh" "${args[@]}" "$quota" 50 "$data" "$logs" /left_wrist_camera/color/image_raw none peach
