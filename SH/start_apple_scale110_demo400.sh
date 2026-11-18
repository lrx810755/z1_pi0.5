#!/usr/bin/env bash
set -euo pipefail
root=/home/lp531/robot/apple_scale110_demo400_20261105
logs=/home/lp531/z1_queue_logs/apple_scale110_demo400_20261105
data=/home/lp531/smolvla_data/apple_scale110_demo400_20261105
export ROS_DOMAIN_ID=130
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export Z1_ISAAC_GRASP_CONFIG="$root/grasp_physics.json"
export Z1_COLLECTION_GRID_ROWS=5 Z1_COLLECTION_GRID_COLUMNS=4
case "${1:-}" in
 --status) systemctl --user status apple-scale110-demo400-20261105.service --no-pager; exit;;
 ''|--dry-run) ;;
 *) echo '支持 --dry-run、--status 或无参数启动';exit 2;;
esac
mkdir -p "$logs"
cp "$root/collection.launch.py" "$logs/collection.launch.py"
if [[ "${1:-}" != --dry-run ]];then
 exec 9>"$logs/.collection.lock"
 flock -n 9 || { echo '采集入口已运行';exit 1; }
 free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
 if (( free_mib < 18000 ));then echo '显存不足，拒绝启动第二个采集或争用训练显存';exit 1;fi
fi
args=()
if [[ "${1:-}" == --dry-run ]];then args+=(--dry-run);fi
cat "$root/collection_manifest.json"
exec bash "$root/start_collection.sh" "${args[@]}" 20 50 "$data" "$logs" /left_wrist_camera/color/image_raw none apple
