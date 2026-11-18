#!/usr/bin/env bash
# 抓取一键示教失败文本（含 local/target），用于判断超差方向。
OUT="/home/lp531/z1_queue_logs/cup_failures.log"
prev=""
end=$(( $(date +%s) + ${1:-3600} ))
while (( $(date +%s) < end )); do
  msg=$(curl -sS --max-time 2 http://127.0.0.1:8767/api/status 2>/dev/null | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    raise SystemExit
a=d.get('auto_demonstration') or {}
print(json.dumps({'phase':a.get('phase'),'step':a.get('step'),'message':a.get('message')},ensure_ascii=False))
" 2>/dev/null)
  if [[ -n "$msg" && "$msg" != "$prev" ]]; then
    if echo "$msg" | grep -q "失败\|超差\|区间\|error_code"; then
      printf '[%s] %s\n' "$(date '+%F %T')" "$msg" >> "$OUT"
    fi
    prev="$msg"
  fi
  sleep 0.5
done
