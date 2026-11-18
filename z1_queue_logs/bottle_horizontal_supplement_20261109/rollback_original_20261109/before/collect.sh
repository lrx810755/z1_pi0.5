#!/usr/bin/env bash
set -euo pipefail

z1_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
quota="${1:-8}"
max_failures="${2:-50}"
raw_root="${3:-/home/lp531/smolvla_data/z1_tabletop_raw_balanced_three_camera_v1}"
progress_file="${4:-/home/lp531/z1_queue_logs/smolvla_balanced_three_camera_v1/progress.json}"
target_selection="${5:-beverage_bottle}"
if (( $# > 5 )); then echo "参数过多" >&2; exit 2; fi
[[ "$target_selection" == beverage_bottle ]] || { echo "独立入口仅支持 beverage_bottle" >&2; exit 2; }
case "$target_selection" in
  all) targets=(apple beverage_bottle peach cup) ;;
  apple|beverage_bottle|peach|cup) targets=("$target_selection") ;;
  *) echo "物体参数只支持：all、apple、beverage_bottle、peach、cup" >&2; exit 2 ;;
esac
[[ "$quota" =~ ^[0-9]+$ ]] && (( quota > 0 )) \
  && [[ "$max_failures" =~ ^[0-9]+$ ]] && (( max_failures > 0 )) || {
  echo "用法：$0 [每方格每物体成功数，默认 8] [累计失败上限，默认 50]" >&2
  exit 2
}

# shellcheck source=configure_environment.sh
source "/home/lp531/ros2_ws/src/z1_ros2/scripts/configure_environment.sh"
z1_workspace_root="$(cd "$Z1_REPO_ROOT/../.." && pwd -P)"
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$z1_workspace_root/install/setup.bash"
set -u

actual_root="$(/usr/bin/python3 /home/lp531/robot/bottle_horizontal_supplement_20261109/startup.py --parameter dataset_root)"
actual_right_topic="$(/usr/bin/python3 /home/lp531/robot/bottle_horizontal_supplement_20261109/startup.py --parameter global_right_rgb_topic)"
/usr/bin/python3 - "$raw_root" "$actual_root" "$actual_right_topic" <<'PYTHON'
from pathlib import Path
import json, sys
expected, actual = (Path(x).expanduser().resolve() for x in sys.argv[1:3])
if expected != actual or sys.argv[3].strip() != '/global_right_camera/color/image_raw':
    raise SystemExit('当前录制器不是指定目录的三相机配置，拒绝开始5×5采集。')
info=json.loads((expected/'dataset_info.json').read_text())
if set(info.get('image_shapes', {})) != {'wrist','global','global_right'}:
    raise SystemExit('录制目录不是三相机数据集，拒绝计入网格配额。')
PYTHON

grid_rows="${Z1_COLLECTION_GRID_ROWS:-5}"
grid_columns="${Z1_COLLECTION_GRID_COLUMNS:-5}"
if [[ ! "$grid_rows" =~ ^[1-5]$ || ! "$grid_columns" =~ ^[1-5]$ ]]; then
  echo '采集行列数必须位于 1 至 5 之间' >&2; exit 2
fi
round=0
while :; do
  round=$((round + 1))
  retry_pending=0
  printf '[%s] 开始采集第 %d 轮\n' "$(date '+%F %T')" "$round"
for ((row=0; row<grid_rows; row++)); do
  # Include column 0; the assistant receives the exact grid column as a ROS parameter.
  for ((column=0; column<grid_columns; column++)); do
    for target in "${targets[@]}"; do
      if (( column == 4 )) && [[ "$target" == apple || "$target" == peach ]]; then
        printf '[%s] 方格 (%d,%d) %s：列坐标 4 的果实暂不采集，跳过\n' \
          "$(date '+%F %T')" "$column" "$row" "$target"
        continue
      fi
      existing="$(ros2 run z1_isaac_bridge collection_progress \
        --source "$raw_root" --state-file "$progress_file" --quota "$quota" \
        --target "$target" --column "$column" --row "$row")"
      if (( existing >= quota )); then
        printf '[%s] 方格 (%d,%d) %s：已有 %d/%d 条已保存，跳过\n' \
          "$(date '+%F %T')" "$column" "$row" "$target" "$existing" "$quota"
        continue
      fi
      while (( existing < quota )); do
        if [[ "$target" == beverage_bottle ]]; then
          column_ready=false
          for ((parameter_attempt=0; parameter_attempt<10; parameter_attempt++)); do
            if timeout --kill-after=2s 15s /usr/bin/python3 /home/lp531/robot/bottle_horizontal_supplement_20261109/set_collection_column.py "$column"; then
              column_ready=true; break
            fi
            sleep 0.3
          done
          if ! "$column_ready"; then echo '无法设置杯子采集列，拒绝使用错误路径策略' >&2; exit 1; fi
        fi
        printf '[%s] 方格 (%d,%d) %s：准备采集第 %d/%d 条（已有数据已保留）\n' \
          "$(date '+%F %T')" "$column" "$row" "$target" \
          "$((existing + 1))" "$quota"
        set +e
        /usr/bin/python3 "$Z1_REPO_ROOT/z1_isaac_bridge/z1_isaac_bridge/continuous_training.py" \
          --port 8839 --target "$target" --curriculum full --episodes "$((quota - existing))" \
          --target-grid "$column,$row" --max-failures "$max_failures"
        training_status=$?
        set -e
        if (( training_status == 3 )); then
          printf '[%s] 方格 (%d,%d) %s：本轮不可达，下一轮重试\n' \
            "$(date '+%F %T')" "$column" "$row" "$target"
          retry_pending=1
          break
        fi
        if (( training_status == 4 )); then
          printf '[%s] 方格 (%d,%d) %s：累计失败达到 %d 次，跳过本次组合（不标记永久不可达）\n' \
            "$(date '+%F %T')" "$column" "$row" "$target" "$max_failures"
          retry_pending=1
          break
        fi
        if (( training_status != 0 )); then
          exit "$training_status"
        fi
        existing="$(ros2 run z1_isaac_bridge collection_progress \
          --source "$raw_root" --state-file "$progress_file" --quota "$quota" \
          --target "$target" --column "$column" --row "$row")"
        printf '[%s] 保存确认：方格 (%d,%d) %s 当前 %d/%d 条\n' \
          "$(date '+%F %T')" "$column" "$row" "$target" "$existing" "$quota"
      done
    done
  done
done
  if (( retry_pending == 0 )); then
    printf '[%s] 所有计划内方格与物体已达到配额，采集完成\n' "$(date '+%F %T')"
    break
  fi
  printf '[%s] 第 %d 轮结束；10 秒后重试未完成组合\n' \
    "$(date '+%F %T')" "$round"
  sleep 10
done
