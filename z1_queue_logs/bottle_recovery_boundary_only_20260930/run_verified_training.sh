#!/usr/bin/env bash
set -Eeuo pipefail
run_log_root=/home/lp531/z1_queue_logs/bottle_recovery_boundary_only_20260930
run_model_root=/home/lp531/smolvla_training/smolvla_bottle_recovery_boundary_only_batch8_20260930
run_python=/home/lp531/miniconda3/envs/unifolm-vla/bin/python
trap 'printf "failed exit=%s at=%s\n" "$?" "$(date -Is)" > "$run_log_root/status.txt"' ERR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
if [[ -n "${1:-}" ]]; then
  printf 'waiting_for_conversion pid=%s at=%s\n' "$1" "$(date -Is)" > "$run_log_root/status.txt"
  while kill -0 "$1" 2>/dev/null; do sleep 10; done
fi
grep -Fq 'LeRobot dataset ready: /home/lp531/smolvla_data/z1_tabletop_lerobot_bottle_recovery_boundary_only_v1' "$run_log_root/convert.log"
printf 'verifying at=%s\n' "$(date -Is)" > "$run_log_root/status.txt"
"$run_python" /home/lp531/verify_bottle_recovery_boundary_only.py > "$run_log_root/verify.log" 2>&1
printf 'training batch=8 steps=30000 at=%s\n' "$(date -Is)" > "$run_log_root/status.txt"
cd /home/lp531/lerobot
/home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train \
  --config_path="$run_log_root/train_config.json" > "$run_log_root/train.log" 2>&1
"$run_python" - "$run_model_root" "$run_log_root" <<'PY'
import json
import sys
from pathlib import Path
from safetensors import safe_open
root, logs = map(Path, sys.argv[1:])
checkpoint = root/'checkpoints/030000'
assert json.loads((checkpoint/'training_state/training_step.json').read_text())['step'] == 30000
config = json.loads((checkpoint/'pretrained_model/train_config.json').read_text())
assert config['batch_size'] == 8 and config['steps'] == 30000
assert config['dataset']['root'] == '/home/lp531/smolvla_data/z1_tabletop_lerobot_bottle_recovery_boundary_only_v1'
with safe_open(str(checkpoint/'pretrained_model/model.safetensors'), framework='pt') as weights:
    assert len(list(weights.keys())) > 0
assert 'End of training' in (logs/'train.log').read_text()
(logs/'completion.json').write_text(json.dumps({'complete': True, 'batch_size': 8, 'steps': 30000,
    'checkpoint': str(checkpoint/'pretrained_model'), 'dataset': config['dataset']['root']}, indent=2)+'\n')
PY
printf 'complete batch=8 steps=30000 at=%s\n' "$(date -Is)" > "$run_log_root/status.txt"
