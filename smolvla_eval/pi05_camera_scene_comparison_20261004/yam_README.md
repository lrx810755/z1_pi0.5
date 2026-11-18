# YAM red-cap fine-tuning

This integration fine-tunes the official `pi05_base` JAX checkpoint on the
single active YAM arm (six joints plus gripper). It provides matched LoRA and
full-fine-tuning configs.

## Contract

- Raw YAM state/action: seven left-arm values, with gripper `0=closed` and
  `1=open`.
- Model-facing gripper: `0=open` and `1=closed`; `YamInputs` and `YamOutputs`
  own the only polarity conversion.
- Arm joints 0-5 are trained as deltas. Gripper 6 stays absolute.
- Model tensor width remains 32 for official-checkpoint compatibility; policy
  output is sliced back to seven values.
- Camera map: `middle -> base_0_rgb`, `left -> left_wrist_0_rgb`, and
  `right -> right_wrist_0_rgb`.
- Dataset and action cadence: 15 Hz, horizon 15.

## Convert and validate

```bash
export HF_LEROBOT_HOME=/path/to/lerobot
uv run examples/yam/convert_yam_data_to_lerobot.py \
  --raw-dir /path/to/gello_software/data/episodes

uv run scripts/compute_norm_stats.py \
  --config-name pi05_yam_red_cap_lora
```

The converter creates deterministic 90-episode train and 10-episode eval
repositories below `HF_LEROBOT_HOME/npow/`. Both training configs deliberately
load the same statistics produced by the LoRA config.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 examples/yam/train_on_vast.sh \
  pi05_yam_red_cap_lora yam_red_cap_lora_run01

CUDA_VISIBLE_DEVICES=1 examples/yam/train_on_vast.sh \
  pi05_yam_red_cap_full yam_red_cap_full_run01
```

The helper starts a new run with `--overwrite`, but switches to `--resume` when
the checkpoint directory already contains a numeric step. Recovery checkpoints
are written every 500 steps; 2,500-step candidates are retained.

## Compare checkpoints

Use the same sample count and seed for both runs:

```bash
uv run examples/yam/evaluate_yam_open_loop.py \
  --config-name pi05_yam_red_cap_lora \
  --checkpoint-dir checkpoints/pi05_yam_red_cap_lora/yam_red_cap_lora_run01/9999 \
  --output results/lora-10000.json

uv run examples/yam/evaluate_yam_open_loop.py \
  --config-name pi05_yam_red_cap_full \
  --checkpoint-dir checkpoints/pi05_yam_red_cap_full/yam_red_cap_full_run01/9999 \
  --output results/full-10000.json
```

These held-out action errors catch integration failures and help choose robot
rollout candidates; they do not replace guarded task-success rollouts.

## Servo

The matching Servo branch exposes two distinct embodiments so a seven-channel
checkpoint cannot be mistaken for Servo's existing 14-channel YAM checkpoint:

```bash
servo serve pi0.5 \
  --embodiment yam-left-jax-lora \
  --checkpoint /absolute/path/to/checkpoint/step

servo serve pi0.5 \
  --embodiment yam-left-jax-full \
  --checkpoint /absolute/path/to/checkpoint/step
```

Servo clients send cameras as `top`, `left`, and `right`; the manifest maps
`top` to the policy's `middle` camera. Clients send and receive native YAM
gripper values. Do not invert the gripper again in the client or robot driver.
