#!/usr/bin/env bash
set -euo pipefail
root=/home/lp531/robot/cup_horizontal_demo_20261108
logs=/home/lp531/z1_queue_logs/cup_horizontal_demo_20261108
data=/home/lp531/smolvla_data/cup_horizontal_demo_20261108
quota="${Z1_CUP_QUOTA_PER_CELL:-}"
[[ "$quota" =~ ^[1-9][0-9]*$ ]] || { echo '需用 Z1_CUP_QUOTA_PER_CELL 指定每格条数';exit 2; }
case "${1:-}" in ''|--dry-run) ;; *) echo '只支持无参数或 --dry-run';exit 2;;esac
export ROS_DOMAIN_ID=134
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export Z1_ISAAC_GRASP_CONFIG="$root/grasp_physics.json"
export Z1_COLLECTION_GRID_ROWS=4 Z1_COLLECTION_GRID_COLUMNS=5
mkdir -p "$logs"
cp "$root/collection.launch.py" "$logs/collection.launch.py"
if [[ "${1:-}" != --dry-run ]];then
 exec 9>"$logs/.collection.lock"
 flock -n 9 || { echo '杯子入口已在运行';exit 1; }
 free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
 (( free_mib >= 18000 )) || { echo '显存不足或已有其他仿真/训练，未启动';exit 1; }
fi
args=()
if [[ "${1:-}" == --dry-run ]];then args+=(--dry-run);fi
echo "杯子：20个格子，每格${quota}条，总目标$((quota*20))条"
exec bash "$root/start_collection.sh" "${args[@]}" "$quota" 20 "$data" "$logs" /left_wrist_camera/color/image_raw none cup
