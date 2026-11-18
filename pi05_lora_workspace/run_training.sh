#!/usr/bin/env bash
set -euo pipefail
WORK=/home/lp531/pi05_lora_workspace
UV="$WORK/bootstrap/bin/uv"
PY="$WORK/.venv/bin/python"
DATA=/home/lp531/smolvla_data/z1_tabletop_bottle_pi05_lora_v1
cd "$WORK"
export UV_INDEX_URL=https://pypi.org/simple UV_HTTP_TIMEOUT=180
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DOWNLOAD_TIMEOUT=180 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
exec >> "$WORK/background.log" 2>&1
printf '%s\n' 'STAGE: install isolated environment'
if [ -n "${PI05_BOOTSTRAP_PID:-}" ]; then
    while kill -0 "$PI05_BOOTSTRAP_PID" 2>/dev/null; do sleep 2; done
fi
"$UV" pip install --python "$PY" -e "$WORK/lerobot[pi,peft,training]" \
    'torch==2.9.1' 'torchvision==0.24.1' 'torchcodec==0.8.1' sentencepiece protobuf
printf '%s\n' 'STAGE: download pretrained weights'
"$PY" "$WORK/download_base.py"
if [ ! -s "$WORK/weights/tokenizer/tokenizer.model" ]; then
    curl --fail --location --retry 8 --retry-all-errors --max-time 300 \
        https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
        -o "$WORK/weights/tokenizer/tokenizer.model"
fi
printf '%s\n' 'STAGE: convert dataset without frame removal or reencoding'
if ! "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["codebase_version"] != "v3.0")' "$DATA/meta/info.json"; then
    "$PY" -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id lp531/z1_bottle_pi05_lora \
        --root "$DATA" --push-to-hub false
fi
"$PY" "$WORK/finalize_setup.py"
printf '%s\n' 'STAGE: memory and gradient preflight'
MICROBATCH=2
if ! "$PY" "$WORK/train_lora.py" --microbatch 2 --smoke > "$WORK/smoke_batch2.log" 2>&1; then
    if /usr/bin/grep -Eq 'CUDA out of memory|torch.OutOfMemoryError' "$WORK/smoke_batch2.log"; then
        MICROBATCH=1
        "$PY" "$WORK/train_lora.py" --microbatch 1 --smoke > "$WORK/smoke_batch1.log" 2>&1
    else
        cat "$WORK/smoke_batch2.log"
        exit 1
    fi
fi
if /usr/bin/grep -Eq 'Returning model without loading pretrained weights|Could not load state dict' "$WORK/smoke_batch${MICROBATCH}.log"; then
    printf '%s\n' 'ERROR: pretrained loading failed; refusing to train random weights'
    exit 1
fi
printf '%s\n' "STAGE: training; microbatch=$MICROBATCH; effective_batch=8; optimizer_updates=30000"
"$PY" "$WORK/train_lora.py" --microbatch "$MICROBATCH" > "$WORK/training.log" 2>&1
"$PY" "$WORK/select_best.py"
printf '%s\n' 'STAGE: complete'
