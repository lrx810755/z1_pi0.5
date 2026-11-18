#!/usr/bin/env bash
# Evaluate the bottle-only SmolVLA checkpoint in the Isaac scene.
set -eo pipefail

camera_mode=multi
rollout_args=()
while (($#)); do
  case "$1" in
    --camera-mode)
      if (($# < 2)); then
        echo '用法: --camera-mode single|multi' >&2
        exit 2
      fi
      camera_mode=$2
      shift 2
      ;;
    --camera-mode=*) camera_mode=${1#*=}; shift ;;
    *) rollout_args+=("$1"); shift ;;
  esac
done

case "$camera_mode" in
  single)
    checkpoint=/home/lp531/smolvla_training/smolvla_bottle_global_only_batch8_20260929/checkpoints/030000/pretrained_model
    output=/home/lp531/smolvla_eval/bottle_global_only_batch8_30000
    camera_args=()
    ;;
  multi)
    checkpoint=/home/lp531/smolvla_training/smolvla_bottle_left_wrist_batch8_20260928_220746/checkpoints/030000/pretrained_model
    output=/home/lp531/smolvla_eval/bottle_batch8_30000
    camera_args=(--wrist-rgb-topic /left_wrist_camera/color/image_raw)
    ;;
  *) echo '相机模式必须是 single 或 multi' >&2; exit 2 ;;
esac

if pgrep -f '[/]collect_smolvla_balanced_v2.sh|[/]continuous_training.py' >/dev/null; then
  echo '数据采集仍在运行；请先暂停采集，再启动单瓶子 SmolVLA 场景验证。' >&2
  exit 1
fi

source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -u
export PYTHONPATH="/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

scene_pid=''
cleanup() {
  if [[ -n "$scene_pid" ]]; then
    kill -TERM "$scene_pid" 2>/dev/null || true
    wait "$scene_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if ! pgrep -f '[/]run_z1_ros2_scene.py' >/dev/null; then
  # A 0.25 rad/s PhysX limit caused gravity drift even with a fixed Home target.
  # Keep the scene's default limits; rollout still limits each command step.
  ros2 launch z1_isaac_bridge z1_isaac.launch.py \
    rviz:=false task_gui:=false moveit:=false run_smolvla_policy:=false \
    eval_table_friction:=50 eval_arm_max_velocity:=0.0 &
  scene_pid=$!
  echo '正在启动 Isaac 仿真并等待关节状态...'
  timeout 300 ros2 topic echo --once /joint_states sensor_msgs/msg/JointState >/dev/null || {
    echo 'Isaac 仿真未在 300 秒内就绪，已停止本次验证。' >&2
    exit 1
  }
else
  isaac_pid=$(pgrep -f '[p]ython3 .*run_z1_ros2_scene.py' | head -n 1)
  isaac_started=$(date -d "$(ps -p "$isaac_pid" -o lstart=)" +%s)
  scene_updated=$(stat -c %Y /home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py)
  if (( isaac_started < scene_updated )); then
    echo '现有 Isaac 场景在场景代码更新前启动；请先停止旧场景，再重新运行本脚本。' >&2
    exit 1
  fi
  isaac_args=$(ps -p "$isaac_pid" -o args=)
  if [[ $isaac_args != *'--eval-table-friction 50'* ]] && \
     ! { [[ -f /tmp/z1_eval_friction_live.json ]] &&
          grep -q "\"pid\": $isaac_pid" /tmp/z1_eval_friction_live.json; }; then
    echo '现有 Isaac 场景未启用验证高摩擦；请关闭该场景后重新运行本脚本。' >&2
    exit 1
  fi
  if [[ $isaac_args == *'--eval-arm-max-velocity '* ]] && \
     [[ ! $isaac_args =~ --eval-arm-max-velocity[[:space:]]+0(\.0+)?([[:space:]]|$) ]]; then
    echo '现有 Isaac 场景启用了会导致姿态下沉的关节速度限制；请关闭该场景后重新运行本脚本。' >&2
    exit 1
  fi
  echo '检测到已启用高摩擦的 Isaac 仿真，直接复用。'
fi

/home/lp531/miniconda3/envs/unifolm-vla/bin/python \
  -m z1_isaac_bridge.smolvla_sim_rollout \
  --policy-type smolvla \
  --checkpoint "$checkpoint" \
  "${camera_args[@]}" \
  --training-dataset-root /home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_physical_checked_v1 \
  --max-training-target-distance-m 0.03 \
  --reject-blank-cameras \
  --objects beverage_bottle \
  --target-xy-bounds 0.35 0.70 -0.27 0.16 \
  --episodes-per-object 1 \
  --execute-steps 10 \
  --freeze-target-until-close \
  --max-joint-step 0.10 \
  --timeout 90 \
  --output "$output" \
  "${rollout_args[@]}"
