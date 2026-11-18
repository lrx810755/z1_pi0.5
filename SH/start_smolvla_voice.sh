#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
set -euo pipefail

checkpoint="${SMOLVLA_CHECKPOINT:-/home/lp531/smolvla_training/z1_4x100_state13_2cam_trimmed_chunk30_20260921_225434/checkpoints/last/pretrained_model}"
output="${SMOLVLA_VOICE_OUTPUT:-/home/lp531/smolvla_eval/voice}"
log_dir="${SMOLVLA_VOICE_LOG_DIR:-/home/lp531/z1_queue_logs/smolvla_voice}"
mkdir -p "${output}" "${log_dir}"

scene_started_by_script=0
scene_pid=""
backend_pid=""

cleanup() {
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    timeout 2 ros2 topic pub --once /smolvla/voice_command \
      std_msgs/msg/String "data: '{\"command\":\"stop\"}'" \
      >/dev/null 2>&1 || true
    sleep 1
    kill -- "-${backend_pid}" 2>/dev/null || kill "${backend_pid}" 2>/dev/null || true
    wait "${backend_pid}" 2>/dev/null || true
  fi
  if (( scene_started_by_script )); then
    kill -- "-${scene_pid}" 2>/dev/null || kill "${scene_pid}" 2>/dev/null || true
    wait "${scene_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

full_service_ready() {
  timeout 5 ros2 service list 2>/dev/null \
    | grep -Fxq /smolvla/randomize_episode_full
}

if ! full_service_ready; then
  echo "[系统] 未发现 Isaac/ROS 场景，正在启动训练同款场景..."
  setsid ros2 launch z1_isaac_bridge z1_isaac.launch.py \
    rviz:=false task_gui:=false run_smolvla_policy:=false \
    >"${log_dir}/scene.log" 2>&1 &
  scene_pid=$!
  scene_started_by_script=1
  deadline=$((SECONDS + 300))
  until full_service_ready; do
    if ! kill -0 "${scene_pid}" 2>/dev/null; then
      echo "[错误] Isaac/ROS 场景启动失败，日志: ${log_dir}/scene.log" >&2
      tail -n 40 "${log_dir}/scene.log" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[错误] 等待 /smolvla/randomize_episode_full 超时" >&2
      exit 1
    fi
    sleep 1
  done
else
  echo "[系统] 复用当前运行中的 Isaac/ROS 场景。"
fi

if timeout 5 ros2 topic list 2>/dev/null | grep -Fxq /smolvla/voice_status; then
  echo "[错误] 已有 SmolVLA 语音后端运行，请先结束旧进程。" >&2
  exit 1
fi

echo "[系统] 正在加载 SmolVLA: ${checkpoint}"
setsid env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  /home/lp531/miniconda3/envs/unifolm-vla/bin/python \
  -m z1_isaac_bridge.smolvla_voice_backend \
  --checkpoint "${checkpoint}" \
  --episodes-per-object 1 \
  --output "${output}" \
  >"${log_dir}/backend.log" 2>&1 &
backend_pid=$!

deadline=$((SECONDS + 180))
until timeout 5 ros2 topic list 2>/dev/null | grep -Fxq /smolvla/voice_status; do
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    echo "[错误] SmolVLA 后端启动失败，日志: ${log_dir}/backend.log" >&2
    tail -n 60 "${log_dir}/backend.log" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "[错误] 等待 SmolVLA 语音后端超时" >&2
    exit 1
  fi
  sleep 1
done

echo "[系统] 后端已就绪。语音日志: ${log_dir}/backend.log"
set +e
/home/lp531/miniconda3/envs/robot/bin/python \
  /home/lp531/robot/smolvla_voice_frontend.py "$@"
frontend_status=$?
set -e
exit "${frontend_status}"
