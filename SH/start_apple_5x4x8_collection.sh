#!/usr/bin/env bash
set -euo pipefail
# The existing 5x5 scheduler excludes apple column 4: 5 rows x 4 columns.
exec /home/lp531/start_smolvla_balanced_v2_collection.sh \
  "$@" 8 50 \
  /home/lp531/smolvla_data/z1_tabletop_raw_apple_preplanned_three_camera_20261016 \
  /home/lp531/z1_queue_logs/apple_5x4x8_preplanned_20261016 \
  /left_wrist_camera/color/image_raw none apple
