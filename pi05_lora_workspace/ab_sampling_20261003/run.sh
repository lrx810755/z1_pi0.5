#!/usr/bin/env bash
set -euo pipefail
WORK=/home/lp531/pi05_lora_workspace/ab_sampling_20261003
PY=/home/lp531/pi05_lora_workspace/.venv/bin/python
cd "$WORK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
for ARM in uniform weighted; do
    if [[ -f "$WORK/${ARM}_complete.json" ]]; then continue; fi
    if [[ -e "/home/lp531/smolvla_training/pi05_bottle_sampling_ab_20261003/$ARM" ]]; then
        echo "Existing incomplete output for $ARM; preserve it and inspect before restarting." >&2
        exit 1
    fi
    "$PY" "$WORK/train_ab.py" --arm "$ARM" >> "$WORK/${ARM}.log" 2>&1
done
"$PY" - <<'PY'
from pathlib import Path
import json,re
p=Path.cwd(); result={}
for arm in ['uniform','weighted']:
    completion=json.loads((p/f'{arm}_complete.json').read_text())
    completion['offline_validation']=[{'microstep':int(s),'loss':float(v)} for s,v in re.findall(r'step (\d+): eval_loss=([\d.eE+-]+)',(p/f'{arm}.log').read_text())]
    result[arm]=completion
result['interpretation']='Offline loss only; compare physical grasp behavior before selecting a winner.'
(p/'comparison_summary.json').write_text(json.dumps(result,indent=2))
PY
