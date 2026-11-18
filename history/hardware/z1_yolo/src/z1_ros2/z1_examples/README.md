# Z1 Examples

## Optional YOLO target detection

The real-camera GUI and grasp examples default to YOLO detection. The red-pixel detector remains
available from the GUI recognition-mode selector or with `--detector pixel`. The verified
GPU environment on this workstation is the existing `robot` Conda environment, which contains
CUDA PyTorch 2.5.1. Activate it and install the optional requirements with:

```bash
conda activate robot
cd <repo>/src/z1_ros2
python -m pip install --no-deps -r z1_examples/requirements-yolo.txt
```

`--no-deps` is intentional: Ultralytics declares the GUI-enabled `opencv-python` package,
while this PyQt ROS application uses `opencv-python-headless` to avoid loading a second,
conflicting Qt/xcb plugin bundle.

The default model is the COCO-pretrained `yolo11n.pt` and the default target class is
`bottle`. The GUI target-class field remains editable for any other model class. The detector
resolves the bare `yolo11n.pt` from the installed package share (`<prefix>/share/z1_examples/models`)
after a colcon build. Override the lookup directory with `Z1_MODELS_DIR` or pass an absolute
`.pt` path directly in the GUI. Command-line nodes accept `--detector yolo`,
`--yolo-model`, `--yolo-class`, `--yolo-conf`, and `--yolo-device`. The old `--yolo` flag is
kept as an alias for compatibility.

The grasp GUI also needs a gripper-to-camera hand-eye matrix. Point it at your matrix YAML with
`--calibration <file>` or export `Z1_HANDEYE_CALIBRATION=<file>` before launching.

This package contains miscellaneous codes and scripts to test the Z1 manipulator.

### `waypoint_test.py`

[This executable node](z1_examples/waypoint_test.py) assumes that the robot is configured with a [`joint_trajectory_controller`](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html) that uses the position interface.
By calling the executable
``` 
ros2 run z1_examples waypoint_test.py
```
perform some motion to pass through some joint configuration which are specified in the script as numpy arrays.

To properly run the demo, and not having to worry about the _bringup_ of the robot (wether real or in simulation), you may rely on the corresponding launch file which does everything:
```
ros2 launch z1_examples waypoint_test.launch.py sim_ignition:=true
```
By setting `sim_ignition` to false, the same motion will be executed on the real robot and not on the simulator!
