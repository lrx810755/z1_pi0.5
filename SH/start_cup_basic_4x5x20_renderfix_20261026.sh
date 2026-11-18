#!/usr/bin/env bash
set -euo pipefail

run_dir=/home/lp531/z1_queue_logs/cup_basic_4x5x20_renderfix_20261026
data_dir=/home/lp531/smolvla_data/z1_tabletop_raw_cup_basic_4x5x20_renderfix_20261026
export Z1_COLLECTION_GRID_ROWS=4
export Z1_COLLECTION_GRID_COLUMNS=5
export Z1_ISAAC_GRASP_CONFIG="$run_dir/grasp_physics_minus90.json"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == --status ]]; then
  /usr/bin/python3 - "$run_dir" "$data_dir" <<'PY'
import json, sys
from pathlib import Path
run, data = map(Path, sys.argv[1:])
for name in ('launch.json', 'progress.json'):
    path = run / name
    if path.exists():
        print(name, path.read_text())
print('Saved episode directories:', len(list(data.glob('episode_[0-9]*'))))
print('Quota: 4 rows x 5 columns x 20 successes = 400')
PY
  exit 0
fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo '杯子基础示范：4行×5列，每格20条完整成功示范，共400条。'
  echo '选项：--dry-run 预览；--status 查看记录和配额；无参数开始或继续采集。'
  exit 0
fi
if (( $# > 1 )) || { (( $# == 1 )) && [[ "$1" != --dry-run ]]; }; then
  echo '只支持 --dry-run、--status、--help 或无参数。' >&2
  exit 2
fi
if [[ "${1:-}" != --dry-run ]]; then
  exec 9>"$run_dir/.collection.lock"
  flock -n 9 || { echo '本轮杯子采集已经运行，拒绝重复启动。' >&2; exit 1; }
  nvidia-smi -L >"$run_dir/gpu_preflight.log" 2>&1 || {
    cat "$run_dir/gpu_preflight.log" >&2
    echo 'GPU预检失败，尚未开始录制。' >&2
    exit 5
  }
fi
exec bash "$run_dir/start_collection.sh" "$@" \
  20 20 "$data_dir" "$run_dir" /left_wrist_camera/color/image_raw none cup
