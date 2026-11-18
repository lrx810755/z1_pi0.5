#!/usr/bin/env bash
set -euo pipefail

preview=()
usage() {
  echo "用法：$0 [--dry-run] [每方格每物体成功数，默认8] [累计失败上限，默认50] [物体，默认all]"
  echo "物体：beverage_bottle（瓶子）、apple、peach、cup、all（全部）"
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage; exit 0
fi
if [[ "${1:-}" == --dry-run ]]; then preview=(--dry-run); shift; fi
if (( $# > 3 )); then
  usage >&2
  exit 2
fi
exec /home/lp531/start_smolvla_balanced_v2_collection.sh \
  "${preview[@]}" "${1:-8}" "${2:-50}" \
  /home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1 \
  /home/lp531/z1_queue_logs/smolvla_left_wrist_three_camera_v1 \
  /left_wrist_camera/color/image_raw none "${3:-all}"
