# Z1 Examples

## Optional YOLO target detection

The real-camera GUI and grasp examples default to the existing red-pixel detector. The verified
GPU environment on this workstation is the existing `robot` Conda environment, which contains
CUDA PyTorch 2.5.1. Activate it and install the optional requirements with:

```bash
conda activate robot
cd ~/ros2_ws
python -m pip install --no-deps -r src/z1_ros2/z1_examples/requirements-yolo.txt
```

`--no-deps` is intentional: Ultralytics declares the GUI-enabled `opencv-python` package,
while this PyQt ROS application uses `opencv-python-headless` to avoid loading a second,
conflicting Qt/xcb plugin bundle.

The default model is the COCO-pretrained `yolo11n.pt` and the default target class is
`sports ball`. The detector first checks `~/ros2_ws/models/yolo11n.pt`, then lets Ultralytics
download the weight on first use. Set another local `.pt` path in the GUI for offline or
custom-model operation. Command-line nodes accept `--detector yolo`,
`--yolo-model`, `--yolo-class`, `--yolo-conf`, and `--yolo-device`. The old `--yolo` flag is
kept as an alias for compatibility.

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
