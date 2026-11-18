#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法：$0 [--dry-run] [baseline|left|right|high] [成功总条数，默认30] [每次最多尝试次数，默认150]"
  echo '自动接近、对准、闭爪、放置；保留规划与物理稳定检查的等待，不裁剪中间帧。'
}
preview=false
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ "${1:-}" == --dry-run ]]; then preview=true; shift; fi
variant=${1:-baseline}
episodes=${2:-30}
max_attempts=${3:-150}
if (( $# > 3 )) || [[ ! "$variant" =~ ^(baseline|left|right|high)$ ]] \
  || [[ ! "$episodes" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  usage >&2; exit 2
fi
if (( max_attempts < episodes )); then
  echo '最多尝试次数不能小于成功条数。' >&2; exit 2
fi

raw_root="${BOTTLE_CLOSE_PROMPT_DATA_ROOT:-/home/lp531/smolvla_data/z1_tabletop_raw_bottle_close_prompt_${variant}_v1}"
saved=$(/usr/bin/python3 - "$raw_root" <<'PY'
import json, sys
from pathlib import Path
print(sum(json.loads(p.read_text()).get('task_success') is True
          for p in Path(sys.argv[1]).glob('episode_*/episode_summary.json')))
PY
)
remaining=$((episodes - saved))
echo "已保存 $saved/$episodes 条成功轨迹；本次补录 $((remaining > 0 ? remaining : 0)) 条。"
if (( remaining <= 0 )); then exit 0; fi
scene=(ros2 launch z1_isaac_bridge z1_isaac.launch.py
  rviz:=false task_gui:=true run_smolvla_policy:=false
  target_object:=beverage_bottle bottle_recovery_variant:="$variant"
  freeze_target_keep_collisions:=true dataset_root:="$raw_root"
  wrist_rgb_topic:=/left_wrist_camera/color/image_raw vla_wrist_rgb_topic:=none)
collect=(/usr/bin/python3 /home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/continuous_training.py
  --target beverage_bottle --curriculum full --episodes "$remaining" --max-attempts "$max_attempts")
if $preview; then
  printf '%q ' "${scene[@]}"; printf '\n'
  printf '%q ' "${collect[@]}"; printf '\n'
  exit 0
fi

set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
if timeout 5 ros2 service list 2>/dev/null | grep -Fx /smolvla/randomize_episode_full >/dev/null; then
  echo '已有 Isaac 场景，请先关闭旧场景再运行专用补录入口。' >&2; exit 1
fi
log_dir="/home/lp531/z1_queue_logs/bottle_close_prompt_${variant}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$raw_root" "$log_dir"
export PYTHONUNBUFFERED=1
echo "补录模式：$variant；成功目标：$episodes；最多尝试：$max_attempts"
echo "原始数据：$raw_root"
echo "日志：$log_dir"
echo '对准后由自动流程接续闭爪；失败轮丢弃，成功轮保存完整轨迹。'
setsid "${scene[@]}" >"$log_dir/scene.log" 2>&1 &
scene_pid=$!
cleanup() {
  kill -- "-$scene_pid" 2>/dev/null || kill "$scene_pid" 2>/dev/null || true
  wait "$scene_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + 300))
until timeout 5 ros2 service list 2>/dev/null | grep -Fx /smolvla/randomize_episode_full >/dev/null; do
  kill -0 "$scene_pid" 2>/dev/null || { tail -n 80 "$log_dir/scene.log" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "启动超时：$log_dir/scene.log" >&2; exit 1; }
  sleep 1
done
"${collect[@]}" 2>&1 | tee "$log_dir/collection.log"
