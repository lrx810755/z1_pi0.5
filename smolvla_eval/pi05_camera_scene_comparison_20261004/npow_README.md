---
license: mit
tags:
- robotics
- gello
- yam
- teleoperation
- imitation-learning
- manipulation
- embodied-ai
size_categories:
- 1K<n<10K
---

# YAM GELLO Red Cap Manipulation Dataset (100 Demonstrations)

[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/npow/yam-gello-red-cap-100)

This dataset contains **100 high-quality bimanual teleoperation demonstrations** of a tabletop object grasping and manipulation task using the **I2RT YAM** robot arm teleoperated with a **GELLO** passive leader arm.

---

## Dataset Overview

- **Robot**: I2RT YAM manipulator (6-DOF + parallel jaw gripper, DM-series actuators over SocketCAN).
- **Teleoperation Interface**: GELLO leader arm (Dynamixel XL330-M077 servos, 57,600 baud).
- **Task**: Reaching, grasping, and manipulating a red cylindrical bottle cap / container on a tabletop.
- **Total Demonstrations**: **100 episodes** (`episode_0001` through `episode_0100`).
- **Total Frames**: **25,874 timesteps** (~20-30 Hz control loop).
- **Total Active Duration**: ~22 minutes of continuous teleoperation.
- **Camera Observations**: 3 synchronized RGB cameras (640x480 H.264 @ 15-30 fps):
  - `left.mp4`: Left arm wrist camera (egocentric perspective)
  - `middle.mp4`: Overhead workspace camera (allocentric perspective)
  - `right.mp4`: Right arm wrist camera
- **Proprioception & Action Data** (`trajectory.npz`):
  - `timestamp`: High-precision float64 epoch timestamps
  - `follower_joints`: Real-time joint positions from CAN bus (14 DOFs: 6 left + 1 gripper, 6 right + 1 gripper)
  - `leader_actions`: Commanded joint targets from the GELLO leader
  - `tracking_error`: Absolute tracking error between leader and follower
- **Interactive Visualization**: Native `recording.rrd` streams included per episode for instant inspection in the [Rerun Viewer](https://rerun.io/).

---

## Quality Metrics & Statistics

All 100 episodes passed automated data quality validation:
- **Episode Duration**: Min 9.6s, Median 12.4s, Max 24.3s (0 abandoned or accidental rollouts).
- **Tracking Error**: Average mean tracking error = 0.004 rad (0.2°). Zero joint tracking dropouts.
- **Gripper Actuation**: 100% (100/100) of episodes feature full open-and-close grasp cycles.
- **Camera Sync**: 300/300 video streams decoded with 1:1 frame synchronization to trajectory timesteps.

---

## Directory Structure

```
data/episodes/
├── episode_0001/
│   ├── left.mp4            # Left wrist camera (640x480)
│   ├── middle.mp4          # Overhead workspace camera (640x480)
│   ├── right.mp4           # Right wrist camera (640x480)
│   ├── trajectory.npz      # Joint states, leader actions, timestamps
│   ├── metadata.json       # Episode metadata (num_frames, date)
│   └── recording.rrd       # Rerun native recording stream
├── ...
└── episode_0100/
```

---

## Quick Start & Usage

### 1. Download Dataset with `huggingface_hub`
```bash
huggingface-cli download npow/yam-gello-red-cap-100 --repo-type dataset --local-dir ./yam-red-cap
```

### 2. Loading Trajectory Data in Python
```python
import numpy as np

data = np.load("data/episodes/episode_0100/trajectory.npz")
print("Keys:", list(data.keys()))
print("Timesteps:", len(data["timestamp"]))
print("Follower Joints shape:", data["follower_joints"].shape) # (N, 14)
print("Leader Actions shape:", data["leader_actions"].shape)   # (N, 14)
```

### 3. Interactive Visualization with Rerun
```bash
pip install rerun-sdk
rerun data/episodes/episode_0100/recording.rrd
```

---

## License & Citation
- Software based on [GELLO](https://github.com/wuphilipp/gello_software).
- Hardware based on [I2RT YAM](https://github.com/i2rt-robotics/i2rt) and [GELLO Mechanical](https://github.com/wuphilipp/gello_mechanical).
- Dataset released under the **MIT License**.
