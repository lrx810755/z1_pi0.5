---
license: apache-2.0
tags:
  - robotics
  - vla
  - openpi
  - pi05
  - forcesight
  - tactile
---
# ForceSight — pi0.5 fine-tuned checkpoints
Fine-tuned pi0.5 (openpi) policies for ForceSight manipulation tasks.
All checkpoints are from step 20000, trained on 6/3 data.
## Checkpoints
| Folder | Variant | Description |
|--------|---------|-------------|
| `pi05_6_3/` | Baseline pi0.5 | Vanilla pi0.5 fine-tune, no tactile. |
| `encoder_6_3/` | pi0.5 + tactile encoder | Conv-Based encoder for tactile images |
| `tactile_6_3/` | pi0.5 + tactile | Tactile images are augmented as camera inputs to the VLA model |
| `tapvla_6_3/` | pi0.5 + annotation | Tactile sensor data is annotated directly on the VLA images |
Each folder contains `params/` (orbax weights) and `assets/` (normalization stats — required for inference).
## Setup
- **Base model:** pi0.5 (openpi)
- **Robot:** Franka Emika Panda + Franka Hand
- **Tasks:** Medicine, Balance, Gear Insertion, Plug Insertion. 
- **Training:** 20000 steps, 4 A6000 GPUs.
## Loading
Download a single checkpoint:
```bash
hf download mlshehab/forcesight --include "pi05_6_3/*" --local-dir ./forcesight
```
Load with openpi:
```python
from openpi.policies import policy_config
from openpi.training import config
cfg = config.get_config("<FILL IN: config name, e.g. pi05_forcesight>")
policy = policy_config.create_trained_policy(cfg, "./forcesight/pi05_6_3")
```
> Note: the tactile and TAP-VLA variants require a custom openpi config/fork. See [openpi](https://github.com/Physical-Intelligence/openpi).
