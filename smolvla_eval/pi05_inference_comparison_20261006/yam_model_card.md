---
license: apache-2.0
library_name: openpi
tags:
  - robotics
  - vision-language-action
  - pi0.5
  - openpi
  - yam
  - servo
---

# pi0.5 YAM red-cap full fine-tune — step 7,500

This repository contains a full-model fine-tune of Physical Intelligence's official pi0.5 base checkpoint for a single-active-arm YAM/GELLO task: pick up a red cap and place it in a black box.

This is an OpenPI JAX/Orbax checkpoint. It is not a Transformers checkpoint and cannot be loaded with `AutoModel.from_pretrained`.

## Model and interface

- Base: official `gs://openpi-assets/checkpoints/pi05_base/params`
- Training method: full fine-tuning; no LoRA and no frozen model weights
- Checkpoint: training step 7,500
- OpenPI source: [`npow/openpi`, `feat/yam-redcap-finetune`](https://github.com/npow/openpi/tree/feat/yam-redcap-finetune)
- OpenPI base commit: `215abfb217dbac7d5f1273282331b9b1866c0479`
- YAM fine-tuning pipeline commit: `8539331ec241dfa3b232fc792115548348567bc5`
- Model tensor width: 32 state/action dimensions, with OpenPI padding the 7 physical dimensions
- Robot-facing state/action: 7 values (`6` arm joints + `1` gripper)
- Action chunk: `15 x 7`, representing one second at 15 Hz
- Cameras: three RGB views, consistently mapped as `middle -> base_0_rgb`, `left -> left_wrist_0_rgb`, and `right -> right_wrist_0_rgb`
- Native YAM and Servo gripper convention: `0 = closed`, `1 = open`

The YAM transform flips the gripper to pi0.5's model-facing convention before normalization and flips it back on output. Joint channels 0–5 are trained as deltas; the gripper channel remains absolute.

The seven-dimensional policy interface is side-neutral. The deployment binding selects whether those values control the physical left or right arm; the inactive arm must be held by the robot controller.

## Training data

The source collection is public: [`npow/yam-gello-red-cap-100`](https://huggingface.co/datasets/npow/yam-gello-red-cap-100).

It contains 100 demonstrations and three synchronized camera streams. The training conversion resampled trajectory timestamps to 15 Hz, selected the active left-arm dimensions, and used a deterministic episode split:

- Training: 90 episodes / 17,564 frames
- Held out: 10 episodes / 1,993 frames
- Held-out source episode IDs: `6, 16, 26, 36, 46, 56, 66, 76, 86, 96`
- Prompt: `pick up the red cap and place it in the black box`

The exact converter, transforms, and training configuration are in the [YAM example on the training branch](https://github.com/npow/openpi/tree/feat/yam-redcap-finetune/examples/yam).

## Training configuration

- Model: `Pi0Config(pi05=True, action_dim=32, action_horizon=15)`
- Batch size: 32
- LR: cosine decay, 500 warm-up steps, `2.5e-5` peak, `2.5e-6` final at 10,000 steps
- EMA: disabled
- Seed: 42
- Initialization: official pi0.5 base only

Normalization statistics from the converted training split are bundled under `assets/`.

## Download and serve

```bash
hf download npow/pi05-yam-red-cap-full-7500 \
  --local-dir ./pi05-yam-red-cap-full-7500

servo serve pi0.5 \
  --embodiment yam-single \
  --checkpoint "$PWD/pi05-yam-red-cap-full-7500"
```

The intended deployment is through Servo's `yam-single` contract with `top`, `left`, and `right` camera roles. Servo uses the same 7D policy shape for left- or right-arm placement; arm selection belongs to the robot binding, not the checkpoint.

## Limitations and safety

This checkpoint is specialized to one tabletop task, one robot setup, one camera arrangement, and a small demonstration set. It should not be treated as a general-purpose manipulation policy. Camera role/order, state order, action rate, normalization statistics, and gripper semantics must match training.

Robot execution requires independent joint/rate/workspace limits, timeout-to-hold behavior, an inactive-arm hold controller, reduced-speed first rollouts, and a reachable emergency stop. The model itself is not a safety controller.
