#!/usr/bin/env bash
set -euo pipefail
cd /home/lp531/pi05_lora_workspace/pi05_cup_recovery513_lora_20261029
exec 9>.launch.lock
flock -n 9
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
unit=pi05-cup-recovery513-lora-20261029
if systemctl --user is-active --quiet "$unit.service" 2>/dev/null; then
    echo "Training service already active: $unit.service"
    exit 0
fi
/home/lp531/pi05_lora_workspace/.venv/bin/python -u runner.py --preflight
if systemctl --user show-environment >/dev/null 2>&1; then
    systemd-run --user --unit="$unit" --collect \
        --property=WorkingDirectory="$PWD" --property=KillMode=control-group \
        /bin/bash "$PWD/run_training.sh"
else
    nohup /bin/bash "$PWD/run_training.sh" </dev/null > launch_stdout.log 2>&1 9>&- &
    echo "$!" > workflow.pid
    echo "Submitted workflow PID $(cat workflow.pid); progress evidence is written by runner."
fi
