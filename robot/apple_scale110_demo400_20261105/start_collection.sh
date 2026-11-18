#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "用法：$0 [--dry-run] [每方格每物体成功数，默认8] [累计失败上限，默认50] [数据目录] [日志目录] [腕部话题] [none] [物体，默认all]"
  echo "物体：beverage_bottle（瓶子）、apple、peach、cup、all（全部）"
}
preview=false
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ "${1:-}" == --dry-run ]]; then preview=true; shift; fi
raw_root="${3:-/home/lp531/smolvla_data/z1_tabletop_raw_balanced_three_camera_v1}"
quota="${1:-8}"
max_attempts="${2:-50}"
log_dir="${4:-/home/lp531/z1_queue_logs/smolvla_balanced_three_camera_v1}"
wrist_rgb_topic="${5-/left_wrist_camera/color/image_raw}"
vla_wrist_rgb_topic="${6-none}"
target_selection="${7:-all}"
if (( $# > 7 )) || [[ ! "$quota" =~ ^[1-9][0-9]*$ || ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  usage >&2; exit 2
fi
case "$target_selection" in
  all) initial_target=apple ;;
  apple|beverage_bottle|peach|cup) initial_target="$target_selection" ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "$vla_wrist_rgb_topic" != none && -n "$vla_wrist_rgb_topic" ]]; then
  echo '三相机采用腕部与两个外部视角；第六参数请使用 none。' >&2; exit 2
fi
/usr/bin/python3 - "$raw_root" <<'PYTHON'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); info=root/'dataset_info.json'
if info.exists():
    d=json.loads(info.read_text())
    expected={'wrist':[360,640,3], 'global':[480,640,3], 'global_right':[480,640,3]}
    if d.get('image_shapes') != expected:
        raise SystemExit('该目录不是当前三相机数据集，请使用新目录；旧数据不会被覆盖。')
elif any(root.glob('episode_*')) or any(root.glob('.recording_*')):
    raise SystemExit('该目录有回合但缺少三相机元数据，请使用新目录。')
PYTHON
progress_file="$log_dir/progress.json"
scene=(ros2 launch "$log_dir/collection.launch.py"
  task_gui_port:=8833 external_camera_layout:=dual headless:=false rviz:=false task_gui:=true run_smolvla_policy:=false
  freeze_target_keep_collisions:=true dataset_root:="$raw_root" target_object:="$initial_target"
  wrist_rgb_topic:="$wrist_rgb_topic" vla_wrist_rgb_topic:="$vla_wrist_rgb_topic"
  recording_fps:=10.0 recording_sync_tolerance_s:=0.045
  recording_sync_delay_s:=0.5 recording_action_age_s:=0.40)
collect=(/home/lp531/robot/apple_scale110_demo400_20261105/collect.sh
  "$quota" "$max_attempts" "$raw_root" "$progress_file" "$target_selection")
if $preview; then
  printf '%q ' "${scene[@]}"; printf '\n'
  printf '%q ' "${collect[@]}"; printf '\n'
  exit 0
fi
mkdir -p "$raw_root" "$log_dir"
export PYTHONUNBUFFERED=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/fastdds_camera.xml
export FASTDDS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE"

set +u
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u

if timeout 5 ros2 service list 2>/dev/null | grep -Fx /smolvla/randomize_episode_full >/dev/null; then
  echo "检测到已有 Isaac 场景；无法确认其 dataset_root。请先关闭旧场景后重新运行。" >&2
  exit 1
fi

echo "4列×5行 三相机采集；物体：$target_selection；每格每物体配额：$quota；数据：$raw_root"
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

/usr/bin/python3 /home/lp531/robot/apple_scale110_demo400_20261105/startup.py --timeout 300 >"$log_dir/startup_verified.json"
"${collect[@]}" 2>&1 | tee -a "$log_dir/collection.log"
