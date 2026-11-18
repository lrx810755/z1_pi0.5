#!/usr/bin/env bash
# 守护：等 peach 采集结束（数据集不再增长）后，重启整个训练栈，并排队 cup:full:100。
# 触发条件（任一，且最后一条 Episode 的 target_object 必须是 peach）：
#   1) peach 累计条数达到 PEACH_TARGET_COUNT（默认 100）
#   2) 数据集停顿 >= STALL_S（默认 1800s）且已无 continuous_training 进程
#   3) 数据集停顿 >= STALL_FORCE_S（默认 3600s，兜底：采集卡死）
# 用法：nohup setsid bash watch_restart_peach_to_cup.sh >/dev/null 2>&1 &
set -uo pipefail

HOME_DIR="${HOME:-/home/lp531}"
DATASET_JSONL="${DATASET_JSONL:-$HOME_DIR/smolvla_data/z1_tabletop_raw_dual_wrist/episodes.jsonl}"
LOG_DIR="${LOG_DIR:-$HOME_DIR/z1_queue_logs}"
POLL_S="${POLL_S:-60}"
STALL_S="${STALL_S:-1800}"
STALL_FORCE_S="${STALL_FORCE_S:-3600}"
PEACH_TARGET_COUNT="${PEACH_TARGET_COUNT:-100}"
NEXT_JOB="${NEXT_JOB:-cup:full:100}"
WAIT_STACK_S="${WAIT_STACK_S:-900}"
DRY_RUN="${WATCH_DRY_RUN:-0}"

Z1_WS="$HOME_DIR/ros2_ws"
Z1_SCRIPTS="$Z1_WS/src/z1_ros2/scripts"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/watch_peach_to_cup_$(date '+%Y%m%d_%H%M%S').log"
DONE_FILE="$LOG_DIR/watch_peach_to_cup.done"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$LOG_FILE"; }

count_total() { grep -c . "$DATASET_JSONL" 2>/dev/null || echo 0; }
count_peach() { grep -c '"target_object": "peach"' "$DATASET_JSONL" 2>/dev/null || echo 0; }
last_object() {
  tail -n 1 "$DATASET_JSONL" 2>/dev/null \
    | grep -o '"target_object": "[a-z_]*"' | head -1 | sed 's/.*: "//; s/"$//'
}
collection_running() { pgrep -f 'continuous_training' >/dev/null 2>&1 && return 0 || return 1; }

log "守护启动 PID=$$ 数据集=$DATASET_JSONL 目标pc=$PEACH_TARGET_COUNT 停顿阈值=${STALL_S}s/强制=${STALL_FORCE_S}s 下一任务=$NEXT_JOB"

prev_count=-1
last_change_epoch=$(date +%s)
trigger=""

while true; do
  now=$(date +%s)
  count=$(count_total); peach=$(count_peach); last_obj=$(last_object)
  if [[ "$count" != "$prev_count" ]]; then
    last_change_epoch=$now
    log "数据集增长：总 $prev_count -> $count 条（peach 累计 $peach，最后一条 object=$last_obj）"
    prev_count=$count
  fi
  stall=$(( now - last_change_epoch ))

  if [[ "$last_obj" != "peach" ]]; then
    log "状态：总 $count / peach $peach / 停顿 ${stall}s —— 最后一条不是 peach（$last_obj），不动作"
  else
    if (( peach >= PEACH_TARGET_COUNT )) && [[ "$count" == "$prev_count" ]] && (( stall >= POLL_S )); then
      trigger="peach 已累计 $peach 条（>= $PEACH_TARGET_COUNT），采集结束"
    elif (( stall >= STALL_FORCE_S )); then
      trigger="数据集停顿 ${stall}s（>= ${STALL_FORCE_S}s 兜底阈值）"
    elif (( stall >= STALL_S )) && ! collection_running; then
      trigger="数据集停顿 ${stall}s（>= ${STALL_S}s）且无 continuous_training 进程"
    else
      log "状态：总 $count / peach $peach / 停顿 ${stall}s / 采集进程 $(collection_running && echo 在跑 || echo 已退出)"
    fi
  fi

  [[ -n "$trigger" ]] && break
  sleep "$POLL_S"
done

log "触发条件满足：$trigger"
sleep 20
count_now=$(count_total); last_obj_now=$(last_object)
if [[ "$last_obj_now" != "peach" ]]; then
  log "触发后复查：最后一条变成 $last_obj_now，放弃本次动作并退出"
  exit 0
fi
log "触发后复查：总 $count_now 条，最后一条 = $last_obj_now"

Z1_PIDS_PATTERN='ros2 launch z1_isaac_bridge z1_isaac.launch.py|z1_isaac_bridge/lib/z1_isaac_bridge|moveit_ros_move_group/move_group|run_z1_ros2_scene.py|isaac-sim/python.sh|isaac-sim/kit/python|continuous_training'
z1_list_pids() {
  pgrep -f "$Z1_PIDS_PATTERN" 2>/dev/null | grep -v "^$$\$" || true
}

old_launch_pid=$(pgrep -f 'ros2 launch z1_isaac_bridge z1_isaac.launch.py' 2>/dev/null | head -1)
launch_args=""
declare -A old_env=()
if [[ -n "$old_launch_pid" ]]; then
  old_cmd=$(tr '\0' ' ' <"/proc/$old_launch_pid/cmdline" 2>/dev/null || true)
  log "旧栈：PID $old_launch_pid 命令=$old_cmd"
  launch_args="${old_cmd#*ros2 launch }"
  launch_args="${launch_args% }"
  for key in RMW_IMPLEMENTATION FASTRTPS_DEFAULT_PROFILES_FILE Z1_DATASET_ROOT Z1_ASSET_ROOT Z1_REPO_ROOT ISAAC_SIM_ROOT PYTHONNOUSERSITE DISPLAY WAYLAND_DISPLAY XDG_RUNTIME_DIR; do
    val=$(tr '\0' '\n' <"/proc/$old_launch_pid/environ" 2>/dev/null | grep "^$key=" | head -1 || true)
    [[ -n "$val" ]] && old_env["$key"]="${val#*=}"
  done
else
  launch_args="z1_isaac_bridge z1_isaac.launch.py"
  log "未找到正在运行的 launch，按默认参数重启"
fi
[[ -z "$launch_args" ]] && launch_args="z1_isaac_bridge z1_isaac.launch.py"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY-RUN：将执行停止 -> 重启（ros2 launch $launch_args）-> 排队 $NEXT_JOB"
  touch "$DONE_FILE"
  exit 0
fi

log "步骤1/3：停止训练栈"
kill -INT $old_launch_pid 2>/dev/null || true
for pid in $(z1_list_pids); do kill -INT "$pid" 2>/dev/null || true; done
for round in 1 2 3; do
  sleep 15
  remaining=$(z1_list_pids)
  [[ -z "$remaining" ]] && break
  log "第 $round 轮等待，仍存活：$(echo $remaining | tr '\n' ' ')"
  if (( round == 1 )); then for pid in $remaining; do kill -TERM "$pid" 2>/dev/null || true; done; fi
done
remaining=$(z1_list_pids)
if [[ -n "$remaining" ]]; then
  log "强制结束残留进程：$(echo $remaining | tr '\n' ' ')"
  for pid in $remaining; do kill -KILL "$pid" 2>/dev/null || true; done
  sleep 5
fi
log "残留进程：$(z1_list_pids | tr '\n' ' ' | sed 's/^$/(无)/')"

log "步骤2/3：重启训练栈：ros2 launch $launch_args"
stack_log="$LOG_DIR/stack_$(date '+%Y%m%d_%H%M%S').log"
cd "$Z1_WS" || exit 1
env_args=()
for key in "${!old_env[@]}"; do env_args+=("$key=${old_env[$key]}"); done
setsid nohup env "${env_args[@]}" \
  bash -lc "cd '$Z1_WS' && source '$Z1_SCRIPTS/configure_environment.sh' && source /opt/ros/humble/setup.bash && source '$Z1_WS/install/setup.bash' && exec ros2 launch $launch_args" \
  >>"$stack_log" 2>&1 </dev/null &
new_pid=$!
log "已后台启动新栈 PID $new_pid，日志 $stack_log"

deadline=$(( $(date +%s) + WAIT_STACK_S ))
api_ok=0
while (( $(date +%s) < deadline )); do
  if curl -sS --max-time 3 http://127.0.0.1:8767/api/status >/dev/null 2>&1; then api_ok=1; break; fi
  sleep 20
done
if (( api_ok == 1 )); then log "GUI/API 8767 已就绪"; else log "警告：${WAIT_STACK_S}s 内 8767 未就绪，仍继续排队"; fi
sleep 30

log "步骤3/3：排队 $NEXT_JOB"
cd "$Z1_SCRIPTS" || exit 1
if bash "$Z1_SCRIPTS/queue_continuous_training.sh" --detach --job "$NEXT_JOB" >>"$LOG_FILE" 2>&1; then
  log "已排队 $NEXT_JOB"
else
  log "排队 $NEXT_JOB 失败，请手动执行"
fi
touch "$DONE_FILE"
log "守护结束"
