#!/usr/bin/env bash
set -euo pipefail
root=/home/lp531/robot/cup_horizontal_demo_20261105
logs=/home/lp531/z1_queue_logs/cup_horizontal_v3_demo_20261105
data=/home/lp531/smolvla_data/cup_horizontal_v3_demo_20261105
mode="${1:-}"
case "$mode" in
 --help|-h) echo '独立杯子连续水平接近示范：从第二列开始，共16格，每格1条，10Hz三相机。--dry-run预览；--status查看；无参数采集。'; exit 0;;
 --status) /usr/bin/python3 - "$data" "$logs" <<'PY'
from pathlib import Path
import sys
print('Saved:',len(list(Path(sys.argv[1]).glob('episode_*/episode_summary.json'))))
p=Path(sys.argv[2])/'progress.json'
if p.exists():print(p.read_text())
PY
 exit 0;;
 ''|--dry-run) ;;
 *) echo '只支持 --dry-run / --status / --help 或无参数' >&2;exit 2;;
esac
if (( $# > 1 ));then echo '参数过多' >&2;exit 2;fi
mkdir -p "$logs"
cp "$root/collection.launch.py" "$logs/collection.launch.py"
export ROS_DOMAIN_ID=129
export Z1_SERVO_PREFIX=/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble
export Z1_ISAAC_GRASP_CONFIG="$root/grasp_physics.json"
export Z1_COLLECTION_GRID_ROWS=4 Z1_COLLECTION_GRID_COLUMNS=5
if [[ "$mode" != --dry-run ]];then
 exec 9>"$logs/.collection.lock"
 flock -n 9 || { echo '此录制入口已在运行';exit 1; }
 free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
 if (( free_mib < 18000 ));then echo 'GPU正在使用，拒绝与训练/评估争用显存；未启动采集。' >&2;exit 1;fi
fi
args=()
if [[ "$mode" == --dry-run ]];then args+=(--dry-run);fi
/usr/bin/python3 - "$root" <<'PY'
import json,sys
sys.path.insert(0,sys.argv[1])
from core import Settings
from dataclasses import asdict
s=Settings();print(json.dumps(dict(target='cup',settings=asdict(s),final_velocity_scaling=s.final_speed,waypoint_stops=False,segments="2 first, 3 on fallback",retreat_candidates_mm=[round(x*1000) for x in s.retreat_candidates_m],high_clearance_candidates_mm=[80,120],high_azimuth_candidates_deg=[0,15,-15],final_pose_stop_for_close=True,frames_per_second=10,start_column=1,columns=[1,2,3,4],rows=[0,1,2,3],quota=16,implementation='Low-first PTP+LIN; via-high PTP+LIN+LIN only on rejection; all blended'),ensure_ascii=False,indent=2))
PY
exec bash "$root/start_collection.sh" "${args[@]}" 1 20 "$data" "$logs" /left_wrist_camera/color/image_raw none cup
