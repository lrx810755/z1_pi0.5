#!/usr/bin/env bash
# Queue continuous_training collections behind a trigger.
#
# The wrapper waits for a trigger (a running collection finishing, a wall clock
# time, or a delay), optionally confirms that the previous collection really
# saved every Episode, and then runs the requested collections one after
# another. It is safe to detach, so a queue survives closing the terminal that
# started it.
#
# Examples:
#   # start peach:full:100 right after the running collection is complete
#   scripts/queue_continuous_training.sh --detach \
#     --wait-for continuous_training --wait-min-episodes 102 \
#     --job peach:full:100
#
#   # start cup:full:100 at the next 03:00
#   scripts/queue_continuous_training.sh --detach --at 03:00 \
#     --job cup:full:100
#
#   # print the resolved plan without waiting or running anything
#   scripts/queue_continuous_training.sh --dry-run \
#     --wait-for continuous_training --job peach:full:100
set -euo pipefail

z1_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
z1_dataset_dir_name="z1_tabletop_raw_dual_wrist"
z1_target_objects=(cup apple peach mouse beverage_bottle)
z1_curriculum_levels=(easy medium full)
z1_poll_interval_s=5

z1_usage() {
  cat <<'USAGE'
用法：queue_continuous_training.sh [触发方式] [完整性校验] [运行控制] --job ...

连续采集排队器：等待触发条件后，按顺序执行一个或多个 continuous_training 采集。

采集任务（至少一个，可重复）：
  --job TARGET:CURRICULUM:EPISODES
                         例如 --job peach:full:100
                         TARGET：cup apple peach mouse beverage_bottle
                         CURRICULUM：easy medium full
                         EPISODES：需要成功保存的轮数（不是最大尝试次数）

触发方式（互斥，缺省为 --now）：
  --now                  立即执行（默认）
  --wait-pid PID         等该 PID 退出后再执行
  --wait-for REGEX       等所有匹配 REGEX 的进程退出后再执行
  --at TIME              等到某个时间点执行；支持 HH:MM、HH:MM:SS、
                         "YYYY-MM-DD HH:MM:SS"；HH:MM 已过则顺延到明天
  --delay SECONDS        等待指定秒数后执行

完整性校验：
  --wait-min-episodes N  触发后继续等到数据集 episodes.jsonl 至少 N 条记录
                         才认为上一轮采集完整；超时则放弃排队（不启动采集）
  --wait-timeout SECONDS 上面这个等待的超时，0 表示无限等待（默认 0）
  --episodes-jsonl PATH  覆盖 episodes.jsonl 路径
                         （默认 $Z1_DATASET_ROOT/z1_tabletop_raw_dual_wrist）

运行控制：
  --pattern REGEX        判断“是否已有采集在跑”的进程匹配串，
                         默认 continuous_training
  --no-wait-running      不等待已在运行的采集（默认会等待，避免叠加请求）
  --keep-going           某个采集失败后继续执行后面的采集
  --log-dir DIR          日志与 pid 文件目录，默认 $HOME/z1_queue_logs
  --dry-run              只打印解析后的计划，不等待也不执行
  --detach               后台运行并立即返回（setsid，关闭终端也会继续）
  --status               查看后台队列状态
  --stop                 停止后台队列（向采集进程发送 SIGTERM，优雅收尾）
  -h, --help             显示本帮助

环境变量：
  Z1_QUEUE_LOG_DIR       等价于 --log-dir
  Z1_DATASET_ROOT        数据集根目录，默认 $HOME/smolvla_data
USAGE
}

z1_fail() {
  printf '错误：%s\n' "$*" >&2
  exit 2
}

z1_log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

z1_jobs=()
z1_trigger="now"
z1_trigger_count=0
z1_wait_pid=""
z1_wait_pattern=""
z1_wait_at=""
z1_delay_s=""
z1_wait_min_episodes=""
z1_wait_timeout_s=0
z1_episodes_jsonl=""
z1_pattern="continuous_training"
z1_wait_running=1
z1_keep_going=0
z1_log_dir="${Z1_QUEUE_LOG_DIR:-$HOME/z1_queue_logs}"
z1_dry_run=0
z1_detach=0
z1_action="run"
z1_forward_args=()

z1_set_trigger() {
  z1_trigger_count=$((z1_trigger_count + 1))
  if (( z1_trigger_count > 1 )); then
    z1_fail "触发方式只能指定一个（--now/--wait-pid/--wait-for/--at/--delay）"
  fi
  z1_trigger="$1"
}

while (($#)); do
  case "$1" in
    --job)
      [[ -n "${2:-}" ]] || z1_fail "--job 需要 TARGET:CURRICULUM:EPISODES"
      z1_jobs+=("$2"); z1_forward_args+=("$1" "$2"); shift 2 ;;
    --now)
      z1_set_trigger now; z1_forward_args+=("$1"); shift ;;
    --wait-pid)
      [[ -n "${2:-}" ]] || z1_fail "--wait-pid 需要 PID"
      z1_set_trigger pid; z1_wait_pid="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --wait-for)
      [[ -n "${2:-}" ]] || z1_fail "--wait-for 需要正则表达式"
      z1_set_trigger pattern; z1_wait_pattern="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --at)
      [[ -n "${2:-}" ]] || z1_fail "--at 需要时间"
      z1_set_trigger at; z1_wait_at="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --delay)
      [[ -n "${2:-}" ]] || z1_fail "--delay 需要秒数"
      z1_set_trigger delay; z1_delay_s="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --wait-min-episodes)
      [[ -n "${2:-}" ]] || z1_fail "--wait-min-episodes 需要条数"
      z1_wait_min_episodes="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --wait-timeout)
      [[ -n "${2:-}" ]] || z1_fail "--wait-timeout 需要秒数"
      z1_wait_timeout_s="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --episodes-jsonl)
      [[ -n "${2:-}" ]] || z1_fail "--episodes-jsonl 需要路径"
      z1_episodes_jsonl="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --pattern)
      [[ -n "${2:-}" ]] || z1_fail "--pattern 需要正则表达式"
      z1_pattern="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --no-wait-running)
      z1_wait_running=0; z1_forward_args+=("$1"); shift ;;
    --keep-going)
      z1_keep_going=1; z1_forward_args+=("$1"); shift ;;
    --log-dir)
      [[ -n "${2:-}" ]] || z1_fail "--log-dir 需要路径"
      z1_log_dir="$2"
      z1_forward_args+=("$1" "$2"); shift 2 ;;
    --dry-run)
      z1_dry_run=1; z1_forward_args+=("$1"); shift ;;
    --detach)
      z1_detach=1; shift ;;
    --status)
      z1_action="status"; shift ;;
    --stop)
      z1_action="stop"; shift ;;
    -h|--help)
      z1_usage; exit 0 ;;
    *)
      z1_fail "未知参数：$1（用 --help 查看用法）" ;;
  esac
done

z1_pid_file="$z1_log_dir/queue.pid"

z1_read_pid_file() {
  [[ -f "$z1_pid_file" ]] || return 1
  local z1_text
  z1_text="$(cat "$z1_pid_file" 2>/dev/null || true)"
  [[ "$z1_text" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$z1_text"
}

if [[ "$z1_action" == "status" ]]; then
  if z1_pid="$(z1_read_pid_file)" && kill -0 "$z1_pid" 2>/dev/null; then
    echo "后台队列正在运行：PID $z1_pid"
    ps -o pid,tty,etime,args -p "$z1_pid" 2>/dev/null || true
    echo "日志目录：$z1_log_dir"
    exit 0
  fi
  echo "没有正在运行的后台队列（pid 文件：$z1_pid_file）"
  exit 1
fi

if [[ "$z1_action" == "stop" ]]; then
  if z1_pid="$(z1_read_pid_file)" && kill -0 "$z1_pid" 2>/dev/null; then
    echo "正在停止后台队列 PID $z1_pid（采集进程会优雅收尾）"
    kill -TERM "$z1_pid"
    for _ in $(seq 1 30); do
      kill -0 "$z1_pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$z1_pid" 2>/dev/null && echo "队列仍在退出中，未强杀；可再执行 --stop 或手动 kill"
    exit 0
  fi
  echo "没有正在运行的后台队列（pid 文件：$z1_pid_file）"
  exit 1
fi

for z1_spec in "${z1_jobs[@]}"; do
  IFS=: read -r z1_job_target z1_job_curriculum z1_job_episodes <<<"$z1_spec"
  [[ -n "${z1_job_target:-}" && -n "${z1_job_curriculum:-}" && -n "${z1_job_episodes:-}" ]] \
    || z1_fail "--job 需要 TARGET:CURRICULUM:EPISODES，收到：$z1_spec"
  z1_target_ok=0
  for z1_candidate in "${z1_target_objects[@]}"; do
    [[ "$z1_job_target" == "$z1_candidate" ]] && z1_target_ok=1
  done
  (( z1_target_ok == 1 )) || z1_fail "不支持的目标物体：$z1_job_target"
  z1_curriculum_ok=0
  for z1_candidate in "${z1_curriculum_levels[@]}"; do
    [[ "$z1_job_curriculum" == "$z1_candidate" ]] && z1_curriculum_ok=1
  done
  (( z1_curriculum_ok == 1 )) || z1_fail "不支持的课程等级：$z1_job_curriculum"
  if [[ ! "$z1_job_episodes" =~ ^[0-9]+$ ]] || (( z1_job_episodes <= 0 )); then
    z1_fail "EPISODES 需要正整数，收到：$z1_job_episodes"
  fi
done
if (( ${#z1_jobs[@]} == 0 )); then
  z1_fail "至少需要一个 --job TARGET:CURRICULUM:EPISODES"
fi

if [[ -n "$z1_wait_min_episodes" ]]; then
  if [[ ! "$z1_wait_min_episodes" =~ ^[0-9]+$ ]] || (( z1_wait_min_episodes <= 0 )); then
    z1_fail "--wait-min-episodes 需要正整数，收到：$z1_wait_min_episodes"
  fi
fi
if [[ ! "$z1_wait_timeout_s" =~ ^[0-9]+$ ]]; then
  z1_fail "--wait-timeout 需要非负整数（秒）"
fi
if [[ "$z1_trigger" == "delay" ]] && { [[ ! "$z1_delay_s" =~ ^[0-9]+$ ]] || (( z1_delay_s <= 0 )); }; then
  z1_fail "--delay 需要正整数（秒），收到：$z1_delay_s"
fi

# shellcheck source=configure_environment.sh
source "$z1_script_dir/configure_environment.sh"
z1_workspace_root="$(cd "$Z1_REPO_ROOT/../.." && pwd -P)"

# `ros2 run` needs both the Humble base install and this workspace overlay.
# ROS setup scripts read unset variables, so relax `set -u` while sourcing.
[[ -f /opt/ros/humble/setup.bash ]] \
  || z1_fail "找不到 /opt/ros/humble/setup.bash；请先安装 ROS 2 Humble"
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [[ -f "$z1_workspace_root/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$z1_workspace_root/install/setup.bash"
else
  echo "警告：$z1_workspace_root/install/setup.bash 不存在，工作区可能未构建" >&2
fi
set -u

if [[ -z "$z1_episodes_jsonl" ]]; then
  z1_episodes_jsonl="${Z1_DATASET_ROOT%/}/$z1_dataset_dir_name/episodes.jsonl"
fi

z1_parse_time() {
  local z1_text="$1" z1_epoch
  if [[ "$z1_text" =~ ^[0-9]{1,2}:[0-9]{2}(:[0-9]{2})?$ ]]; then
    z1_epoch="$(date -d "today $z1_text" +%s 2>/dev/null)" || return 1
    if (( z1_epoch <= $(date +%s) )); then
      z1_epoch="$(date -d "tomorrow $z1_text" +%s 2>/dev/null)" || return 1
    fi
  else
    z1_epoch="$(date -d "$z1_text" +%s 2>/dev/null)" || return 1
  fi
  printf '%s\n' "$z1_epoch"
}

z1_target_epoch=""
case "$z1_trigger" in
  at)
    z1_target_epoch="$(z1_parse_time "$z1_wait_at")" \
      || z1_fail "无法解析时间：$z1_wait_at"
    ;;
  delay)
    z1_target_epoch=$(( $(date +%s) + z1_delay_s ))
    ;;
esac

z1_count_episodes() {
  if [[ ! -s "$z1_episodes_jsonl" ]]; then
    echo 0
    return 0
  fi
  local z1_count
  z1_count="$(grep -c . "$z1_episodes_jsonl" 2>/dev/null || true)"
  echo "${z1_count:-0}"
}

z1_read_cmdline() {
  # Empty output means the process is already gone or not readable.
  (tr '\0' ' ' <"/proc/$1/cmdline") 2>/dev/null || true
}

z1_pids_matching() {
  local z1_pattern="$1" z1_pid z1_cmdline
  for z1_pid in $(pgrep -f "$z1_pattern" 2>/dev/null || true); do
    [[ "$z1_pid" == "$$" || "$z1_pid" == "$PPID" ]] && continue
    z1_cmdline="$(z1_read_cmdline "$z1_pid")"
    # A process that disappeared between pgrep and /proc is not running.
    [[ -z "$z1_cmdline" ]] && continue
    # Subshels forked by this script inherit its argv, so they match the
    # pattern too; skip anything that is this queue wrapper itself.
    [[ "$z1_cmdline" == *"$(basename "${BASH_SOURCE[0]}")"* ]] && continue
    echo "$z1_pid"
  done
}

z1_wait_for_pid() {
  local z1_pid="$1" z1_cmdline
  z1_cmdline="$(z1_read_cmdline "$z1_pid")"
  if [[ -n "$z1_cmdline" && "$z1_cmdline" != *"$z1_pattern"* ]]; then
    z1_log "警告：PID $z1_pid 的命令行不含 $z1_pattern，继续等待它退出"
  fi
  local z1_announced=0
  while kill -0 "$z1_pid" 2>/dev/null; do
    if (( z1_announced == 0 )); then
      z1_log "等待 PID $z1_pid 结束：$z1_cmdline"
      z1_announced=1
    fi
    sleep "$z1_poll_interval_s"
  done
  z1_log "PID $z1_pid 已退出"
}

z1_wait_for_pattern() {
  local z1_announced=0
  while :; do
    local z1_pids
    z1_pids="$(z1_pids_matching "$z1_pattern")"
    [[ -z "$z1_pids" ]] && break
    if (( z1_announced == 0 )); then
      z1_log "等待已有采集退出：$(echo "$z1_pids" | tr '\n' ' ')"
      z1_announced=1
    fi
    sleep "$z1_poll_interval_s"
  done
  if (( z1_announced == 1 )); then
    z1_log "已有采集已全部退出"
  fi
}

z1_sleep_until() {
  local z1_until="$1" z1_now z1_remaining
  while :; do
    z1_now="$(date +%s)"
    (( z1_now >= z1_until )) && return 0
    z1_remaining=$(( z1_until - z1_now ))
    z1_log "等待到 $(date -d "@$z1_until" '+%F %T')（剩余 ${z1_remaining}s）"
    if (( z1_remaining > 60 )); then
      sleep 60
    else
      sleep "$z1_remaining"
    fi
  done
}

z1_wait_for_episodes() {
  local z1_required="$1" z1_deadline="" z1_count
  if (( z1_wait_timeout_s > 0 )); then
    z1_deadline=$(( $(date +%s) + z1_wait_timeout_s ))
  fi
  while :; do
    z1_count="$(z1_count_episodes)"
    if (( z1_count >= z1_required )); then
      z1_log "数据集已保存 $z1_count 条 Episode（要求 >= $z1_required），采集完整"
      return 0
    fi
    if [[ -n "$z1_deadline" ]] && (( $(date +%s) >= z1_deadline )); then
      z1_log "等待数据集超时：当前 $z1_count 条，要求 $z1_required 条"
      return 1
    fi
    z1_log "等待数据集补齐：当前 $z1_count/$z1_required 条（$z1_episodes_jsonl）"
    sleep "$z1_poll_interval_s"
  done
}

z1_api_port_open() {
  (exec 3<>/dev/tcp/127.0.0.1/8767) 2>/dev/null
}

z1_child_pid=""

z1_cleanup_pid_file() {
  if [[ -n "${z1_pid_file_written:-}" ]]; then
    rm -f "$z1_pid_file"
  fi
}

z1_on_signal() {
  z1_log "队列收到停止信号，正在停止当前采集"
  if [[ -n "$z1_child_pid" ]] && kill -0 "$z1_child_pid" 2>/dev/null; then
    kill -TERM "$z1_child_pid" 2>/dev/null || true
    wait "$z1_child_pid" 2>/dev/null || true
  fi
  z1_cleanup_pid_file
  exit 130
}

z1_run_job() {
  local z1_spec="$1" z1_rc=0 z1_job_target z1_job_curriculum z1_job_episodes
  IFS=: read -r z1_job_target z1_job_curriculum z1_job_episodes <<<"$z1_spec"
  z1_log "开始采集：target=$z1_job_target curriculum=$z1_job_curriculum episodes=$z1_job_episodes"
  ros2 run z1_isaac_bridge continuous_training \
    --target "$z1_job_target" \
    --curriculum "$z1_job_curriculum" \
    --episodes "$z1_job_episodes" &
  z1_child_pid=$!
  wait "$z1_child_pid" || z1_rc=$?
  z1_child_pid=""
  if (( z1_rc == 0 )); then
    z1_log "采集完成：$z1_spec"
  else
    z1_log "采集退出码 $z1_rc：$z1_spec（130 表示被停止，1 表示异常终止）"
  fi
  return "$z1_rc"
}

z1_describe_trigger() {
  case "$z1_trigger" in
    pid) echo "等 PID $z1_wait_pid 退出" ;;
    pattern) echo "等匹配 '$z1_wait_pattern' 的进程退出" ;;
    at) echo "等到 $(date -d "@$z1_target_epoch" '+%F %T')" ;;
    delay) echo "等待 ${z1_delay_s}s 后（$(date -d "@$z1_target_epoch" '+%F %T')）" ;;
    *) echo "立即开始" ;;
  esac
}

z1_describe_job() {
  local z1_spec="$1" z1_target z1_curriculum z1_episodes
  IFS=: read -r z1_target z1_curriculum z1_episodes <<<"$z1_spec"
  echo "ros2 run z1_isaac_bridge continuous_training --target $z1_target --curriculum $z1_curriculum --episodes $z1_episodes"
}

if (( z1_dry_run == 1 )); then
  echo "触发方式：$(z1_describe_trigger)"
  if [[ -n "$z1_wait_min_episodes" ]]; then
    echo "完整性校验：$z1_episodes_jsonl 至少 $z1_wait_min_episodes 条（当前 $(z1_count_episodes) 条，超时 ${z1_wait_timeout_s}s）"
  fi
  if (( z1_wait_running == 1 )); then
    z1_running_pids="$(z1_pids_matching "$z1_pattern" | tr '\n' ' ')"
    echo "已在运行的采集（启动前会等它们退出）：${z1_running_pids:-（无）}"
  fi
  echo "采集队列："
  for z1_spec in "${z1_jobs[@]}"; do
    echo "  - $(z1_describe_job "$z1_spec")"
  done
  echo "日志目录：$z1_log_dir"
  exit 0
fi

if (( z1_detach == 1 )); then
  mkdir -p "$z1_log_dir"
  z1_log_file="$z1_log_dir/queue_$(date '+%Y%m%d_%H%M%S').log"
  setsid "$0" "${z1_forward_args[@]}" >>"$z1_log_file" 2>&1 &
  z1_spawned=$!
  for _ in $(seq 1 20); do
    if z1_pid="$(z1_read_pid_file)" && kill -0 "$z1_pid" 2>/dev/null; then
      echo "后台队列已启动：PID $z1_pid"
      echo "日志：$z1_log_file"
      echo "查看状态：$0 --status    停止：$0 --stop"
      exit 0
    fi
    kill -0 "$z1_spawned" 2>/dev/null || break
    sleep 0.5
  done
  echo "后台队列已启动（未能确认 PID，spawned=$z1_spawned）"
  echo "日志：$z1_log_file"
  exit 0
fi

command -v ros2 >/dev/null 2>&1 || z1_fail "找不到 ros2；请确认 $z1_workspace_root/install/setup.bash 已构建"

mkdir -p "$z1_log_dir"
z1_pid_file_written=1
printf '%s\n' "$$" >"$z1_pid_file"
trap z1_cleanup_pid_file EXIT
trap z1_on_signal INT TERM

z1_log "队列启动：PID $$，$(z1_describe_trigger)，共 ${#z1_jobs[@]} 个采集任务"
z1_log "日志目录：$z1_log_dir，pid 文件：$z1_pid_file"

case "$z1_trigger" in
  pid) z1_wait_for_pid "$z1_wait_pid" ;;
  pattern) z1_wait_for_pattern ;;
  at|delay) z1_sleep_until "$z1_target_epoch" ;;
esac

if (( z1_wait_running == 1 )); then
  z1_wait_for_pattern
fi

if [[ -n "$z1_wait_min_episodes" ]]; then
  if ! z1_wait_for_episodes "$z1_wait_min_episodes"; then
    z1_fail "上一轮采集未达到 $z1_wait_min_episodes 条 Episode，已放弃排队；确认数据后可用 --now 重新排队"
  fi
fi

if z1_api_port_open; then
  z1_log "采集服务 127.0.0.1:8767 可达"
else
  z1_log "警告：127.0.0.1:8767 暂时不可达，采集脚本会自行等待训练栈就绪"
fi

z1_failed=0
for z1_spec in "${z1_jobs[@]}"; do
  if ! z1_run_job "$z1_spec"; then
    z1_failed=1
    if (( z1_keep_going == 1 )); then
      z1_log "按 --keep-going 继续执行后续采集"
    else
      z1_log "已停止队列；如需失败后继续，请加 --keep-going"
      break
    fi
  fi
done

z1_log "队列结束：数据集现有 $(z1_count_episodes) 条 Episode（$z1_episodes_jsonl）"
exit "$z1_failed"
