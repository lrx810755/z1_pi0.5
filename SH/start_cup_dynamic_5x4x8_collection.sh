#!/usr/bin/env bash
set -euo pipefail
# Same twenty original bottle cells: five columns x four rows, eight successes each.
export Z1_COLLECTION_GRID_ROWS=4
export Z1_COLLECTION_GRID_COLUMNS=5
export Z1_ISAAC_GRASP_CONFIG=/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json
export PYTHONNOUSERSITE=1
if [[ "${1:-}" == --dry-run ]]; then
  printf 'Z1_COLLECTION_GRID_ROWS=4 Z1_COLLECTION_GRID_COLUMNS=5\n'
elif [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo '用法：bash /home/lp531/start_cup_5x4x8_collection.sh [--dry-run]'
  echo '咖啡杯三摄像头动态接近示范；仅接近前冻结，运动前解冻；5列×4行×8条=160条，自动启动场景，循环补齐。'
  exit 0
fi
exec /home/lp531/start_smolvla_balanced_v2_collection.sh \
  "$@" 8 20 \
  /home/lp531/smolvla_data/z1_tabletop_raw_cup_standard_dynamic_three_camera_20261020 \
  /home/lp531/z1_queue_logs/cup_standard_dynamic_5x4x8_20261020 \
  /left_wrist_camera/color/image_raw none cup
