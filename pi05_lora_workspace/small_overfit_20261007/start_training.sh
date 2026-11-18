#!/usr/bin/env bash
set -euo pipefail
unit=pi05-small4-overfit-20261007
if systemctl --user is-active --quiet "$unit.service"; then
  printf '%s\n' 'Small-data training is already running.'
  exit 0
fi
systemd-run --user --unit="$unit" --description='PI05 four-episode overfit experiment, 1000 optimizer updates' \
  --property=WorkingDirectory=/home/lp531/pi05_lora_workspace/small_overfit_20261007 \
  /bin/bash /home/lp531/pi05_lora_workspace/small_overfit_20261007/run_training.sh
