# Z1 dual wrist camera asset

This asset keeps the original hand-eye calibrated D435 and adds a second RGB
camera intended for VLA observations.

## Camera streams

- Original wrist RGB: `/camera/camera/color/image_raw`
- VLA wrist RGB: `/vla_wrist_camera/color/image_raw`
- Global RGB: `/global_camera/color/image_raw`

The added camera uses `vla_camera_color_optical_frame` and a 640x360 D435 RGB
projection. Its bottom-screw mount relative to `gripperStator` is:

```text
xyz = 0.0243802760 -0.0325000000 0.0900367127 m
rpy = 0 0.5597969220 0 rad
```

The original `camera_*` links, calibrated extrinsic, camera prim, and ROS
topics are unchanged.

The VLA mount is shifted 30 mm toward the wrist and 30 mm upward from the
initial visibility-only candidate. This preserves the 32.1 degree pitch while
keeping the D435 body outside the moving finger's complete opening sweep.

## Launch

The standard launch now selects this scene by default:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

## Record all three RGB views

The standard launch records all three RGB views by default into
`$Z1_DATASET_ROOT/z1_tabletop_raw_dual_wrist` (or
`~/smolvla_data/z1_tabletop_raw_dual_wrist`). Existing two-camera
episodes remain compatible and are not modified:

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

The extra frame key is `observation.images.vla_wrist`. The LeRobot exporter
discovers camera keys from `dataset_info.json`, so it supports both the old
two-camera schema and the new three-camera schema.

For a deliberate legacy two-camera recording, override both values and use a
separate compatible dataset root:

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  dataset_root:=$HOME/smolvla_data/z1_tabletop_raw_16x9 \
  vla_wrist_rgb_topic:=""
```

For Windows-native Isaac with ROS in WSL2, follow the split deployment in the
repository's `REPRODUCE.md` and use `scripts/start_ros_stack.sh` in WSL.
