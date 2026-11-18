---
license: apache-2.0
task_categories:
  - robotics
tags:
  - lerobot
  - vla
  - manipulation
  - tactile
  - forcesight
  - franka
configs:
  - config_name: tactile_6_3
    data_files: tactile_6_3/data/**/*.parquet
  - config_name: pi05_6_3
    data_files: pi05_6_3/data/**/*.parquet
  - config_name: tapvla_6_3
    data_files: tapvla_6_3/data/**/*.parquet
---

# ForceSight Demos (6_3)

Teleoperated demonstration datasets in [LeRobot](https://github.com/huggingface/lerobot)
format (v2.1) for the ForceSight manipulation project. Recorded on a Franka Panda
with a parallel-jaw hand and paired vision + tactile sensing.

Companion fine-tuned pi0.5 policies: [mlshehab/forcesight](https://huggingface.co/mlshehab/forcesight).

## Tasks

Four manipulation tasks, teleoperated:

- **balance** — balance a white box on top of blue beams.
- **gear** — insert a gear onto a peg.
- **medicine** — pick up the bottle and place it in the orange bin if full, the blue bin if empty.
- **plug** — insert a plug into a socket.

## Subsets

All three subsets cover the same four tasks (400 episodes, 101,998 frames each).
They differ in observation content / labeling:

| Config | Episodes | Frames | Description |
|--------|----------|--------|-------------|
| `tactile_6_3` | 400 | 101,998 | Full observation set incl. both tactile streams |
| `pi05_6_3` | 400 | 101,998 | pi0.5 baseline (vision + state, no tactile streams) |
| `tapvla_6_3` | 400 | 101,998 | TAP-VLA variant (vision + state, no tactile streams) — <FILL IN: how it differs from pi05_6_3> |

## Format

- **codebase_version:** v2.1
- **robot_type:** panda_hand
- **fps:** 15
- **Note:** image features are stored as `image` (not video-encoded), so parquet
  files are large. Load with the standard LeRobot dataset loader.

### Observations

`tactile_6_3` includes all five image streams below; `pi05_6_3` and `tapvla_6_3`
include only the three camera streams (no `*_touch_image`).

| Key | Type | Shape |
|-----|------|-------|
| `left_shoulder_image` | image | 256×256×3 |
| `right_shoulder_image` | image | 256×256×3 |
| `wrist_image` | image | 256×256×3 |
| `left_touch_image` | image | 240×320×3 |
| `right_touch_image` | image | 240×320×3 |
| `state` | float32 | (8,) joint states |

Tactile images captured with a GelSight sensor.

### Actions

`actions`: float32 (8,) — joint actions.

## Loading

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("mlshehab/forcesight-demos", root="tactile_6_3")
print(ds[0].keys())
```

## License

Apache-2.0.
