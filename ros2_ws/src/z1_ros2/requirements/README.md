# Python dependency groups

The repository separates dependencies by runtime owner:

- `python.txt` contains ordinary Python libraries used by repository tools.
- `yolo.txt` is an optional, isolated object-detection environment.
- `training.txt` is an optional, isolated LeRobot environment.

ROS 2 packages, including `rclpy`, messages, `cv_bridge`, and MoveIt, must be
installed through apt/rosdep. Isaac Sim supplies `isaacsim`, `omni`, `pxr`,
`carb`, and `usdrt`. Do not install those modules from PyPI.

Install the default group with:

```bash
python3 -m pip install -r requirements.txt
```

For YOLO or training, create a separate virtual/Conda environment. Install a
CUDA-compatible Torch build for the target NVIDIA driver before installing
the training group.
