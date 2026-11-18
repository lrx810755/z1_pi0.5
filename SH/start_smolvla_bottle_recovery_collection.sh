#!/usr/bin/env bash
set -euo pipefail

variant="${1:-}"
episodes="${2:-6}"
max_attempts="${3:-30}"
if [[ "$variant" != left && "$variant" != right && "$variant" != high ]] \
  || [[ ! "$episodes" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "用法：$0 {left|right|high} [成功条数，默认 6] [最多尝试次数，默认 30]" >&2
  exit 2
fi

raw_root="/home/lp531/smolvla_data/z1_tabletop_raw_bottle_recovery_${variant}_v1"
log_dir="/home/lp531/z1_queue_logs/smolvla_bottle_recovery_${variant}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$raw_root" "$log_dir"
export PYTHONUNBUFFERED=1

set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u

if timeout 5 ros2 service list 2>/dev/null | grep -Fx /smolvla/randomize_episode_full >/dev/null; then
  echo "检测到已有 Isaac 场景；请先关闭旧场景再运行专用采集入口。" >&2
  exit 1
fi

echo "瓶子纠偏示范：$variant；目标成功数：$episodes；最多尝试：$max_attempts"
echo "原始数据：$raw_root"
echo "运行日志：$log_dir"
setsid ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  rviz:=false task_gui:=true run_smolvla_policy:=false \
  target_object:=beverage_bottle \
  bottle_recovery_variant:="$variant" \
  dataset_root:="$raw_root" \
  wrist_rgb_topic:=/left_wrist_camera/color/image_raw \
  vla_wrist_rgb_topic:=none \
  >"$log_dir/scene.log" 2>&1 &
scene_pid=$!
cleanup() {
  kill -- "-$scene_pid" 2>/dev/null || kill "$scene_pid" 2>/dev/null || true
  wait "$scene_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + 300))
until timeout 5 ros2 service list 2>/dev/null | grep -Fx /smolvla/randomize_episode_full >/dev/null; do
  kill -0 "$scene_pid" 2>/dev/null || {
    tail -n 80 "$log_dir/scene.log" >&2 || true
    exit 1
  }
  (( SECONDS < deadline )) || {
    echo "等待 Isaac 场景超时；日志：$log_dir/scene.log" >&2
    exit 1
  }
  sleep 1
done

python3 /home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/continuous_training.py \
  --target beverage_bottle --curriculum full --episodes "$episodes" \
  --max-attempts "$max_attempts" \
  2>&1 | tee -a "$log_dir/collection.log"
