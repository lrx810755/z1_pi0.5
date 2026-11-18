#!/usr/bin/env bash
set -euo pipefail
root=/home/lp531/robot/dataset_trajectory_review
stamp="$(date +%Y%m%d_%H%M%S)"
output="/home/lp531/z1_queue_logs/dataset_trajectory_review_${stamp}"
unit="dataset-trajectory-review-${stamp}"
mkdir -p "$output"
systemd-run --user --unit="$unit" --property=Type=exec --property=Restart=no \
  --property=Nice=10 --property=KillMode=control-group \
  --property="StandardOutput=append:$output/run.log" \
  --property="StandardError=append:$output/run.log" \
  --setenv=PYTHONNOUSERSITE=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 \
  /usr/bin/python3 -u "$root/review.py" \
  --output "$output"
/usr/bin/python3 - "$output" "$unit" <<'PY'
import json,sys
from pathlib import Path
output,unit=sys.argv[1:]
Path('/home/lp531/z1_queue_logs/dataset_trajectory_review_latest.json').write_text(
    json.dumps(dict(output=output,unit=unit+'.service',monitoring=False),ensure_ascii=False,indent=2))
print('输出目录：'+output)
print('视频：'+str(Path(output)/'异常轨迹汇总_2倍速.mp4'))
print('清单：'+str(Path(output)/'异常候选清单.csv'))
PY
