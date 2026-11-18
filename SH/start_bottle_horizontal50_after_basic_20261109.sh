#!/usr/bin/env bash
set -euo pipefail
root=/home/lp531/robot/bottle_horizontal_supplement_20261109
logs=/home/lp531/z1_queue_logs/bottle_horizontal50_after_basic_20261109
data=/home/lp531/smolvla_data/bottle_horizontal50_after_basic_20261109
preview=false
[[ -f "$root/horizontal50_ready.json" ]] || { echo "40条基础采集尚未完成，拒绝提前启用水平接近" >&2;exit 1; }
if [[ "${1:-}" == --help || "${1:-}" == -h ]];then
 echo '用法：bash start_bottle_horizontal_supplement_20261109.sh [--dry-run] [每格成功条数，默认10；20格共200条]';exit 0
fi
if [[ "${1:-}" == --dry-run ]];then preview=true;shift;fi
quota="${1:-}"
if (( $#>1 )) || [[ ! "$quota" =~ ^[1-9][0-9]*$ ]];then echo '每格成功条数必须是正整数' >&2;exit 2;fi
export ROS_DOMAIN_ID=136
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export Z1_ISAAC_GRASP_CONFIG="$root/grasp_physics.json"
export Z1_COLLECTION_GRID_ROWS=4 Z1_COLLECTION_GRID_COLUMNS=5
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
if ! $preview;then
 mkdir -p "$logs"
 exec 9>"$logs/.collection.lock"
 flock -n 9 || { echo '独立瓶子补采入口已在运行';exit 1; }
 free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
 (( free_mib >= 18000 )) || { echo '可用显存不足18GB，可能有采集/训练正在运行；本次未启动';exit 1; }
fi
args=()
if $preview;then args+=(--dry-run);fi
echo "独立瓶子水平接近：20格，每格${quota}条，总目标$((quota*20))条；ROS域136，端口8839"
exec bash "$root/start_collection.sh" "${args[@]}" "$quota" 20 "$data" "$logs" /left_wrist_camera/color/image_raw none beverage_bottle
