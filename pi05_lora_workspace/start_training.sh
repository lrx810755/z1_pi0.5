#!/usr/bin/env bash
set -euo pipefail
UNIT=pi05-bottle-lora-20261001
if systemctl --user is-active --quiet "$UNIT.service"; then
    printf '%s\n' 'Task is already running'
    exit 0
fi
systemctl --user reset-failed "$UNIT.service" 2>/dev/null || true
PROXY_ARGS=()
for NAME in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
    if [ -n "${!NAME:-}" ]; then PROXY_ARGS+=("--setenv=$NAME=${!NAME}"); fi
done
systemd-run --user --unit=pi05-bottle-lora-20261001 --description='Bottle PI05 LoRA fine-tuning' \
    --property=WorkingDirectory=/home/lp531/pi05_lora_workspace \
    --property="Environment=PI05_BOOTSTRAP_PID=${PI05_BOOTSTRAP_PID:-}" \
    "${PROXY_ARGS[@]}" \
    /bin/bash /home/lp531/pi05_lora_workspace/run_training.sh
