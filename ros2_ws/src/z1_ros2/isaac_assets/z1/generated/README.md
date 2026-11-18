# Z1 SmolVLA multi-object tabletop scene

This directory is generated from:

`/home/lp531/ros2_ws/src/z1_ros2/z1_description/urdf/z1.urdf.xacro`

完整的中文从零复现记录：

`/home/lp531/isaac_assets/z1/generated/REPRODUCTION_GUIDE_2026-07-20.md`

最新工作记录入口：

`/home/lp531/isaac_assets/z1/generated/WORK_LOG_LATEST.md`

2026-07-28 苹果完整重力物理抓取与 ATTACH 门禁专项记录：

`/home/lp531/isaac_assets/z1/generated/WORK_LOG_2026-07-28.md`

## Assets

- `z1_with_d435.urdf`: Xacro-expanded source URDF.
- `z1_with_d435_resolved.urdf`: URDF with ROS package mesh URIs resolved for Isaac Sim.
- `z1_with_d435.usd`: composed Z1 robot asset with wrist RGB and depth cameras.
- `configuration/`: modular visual, physics, robot, and sensor USD layers.
- `smolvla_red_ball_into_box.usd`: multi-object tabletop task scene (legacy filename retained for compatibility).
- `smolvla_red_ball_into_box.json`: task, camera, randomization, and success metadata.
- `wrist_color_preview.png`: runtime smoke-test frame from the wrist RGB camera.
- `../scanned_objects/source/`: downloaded textured USDZ source models.
- `../scanned_objects/usd/`: normalized local USD visual wrappers.
- `../scanned_objects/manifest.json`: dimensions, hashes, collision approximations,
  and source metadata.
- `../scanned_objects/ATTRIBUTION.md`: model creators, licenses, and source links.
- `../scanned_objects/GRASPABILITY_REPORT.md`: gripper dimensions, object
  pinch widths, and recommended grasp regions.

## Camera

The D435 link chain comes directly from the current URDF and the real
ChArUco hand-eye calibration:

- Parent: `gripperStator`
- Bottom-screw mount translation:
  `(0.0900930511, -0.0009311400, -0.0436741355)` m
- Bottom-screw mount RPY:
  `(0.04037701, 0.03561724, 0.01525814)` rad
- Resulting color optical-frame translation:
  `(0.1006887452, 0.0312032005, -0.0302586435)` m
- Resulting color optical-frame quaternion XYZW:
  `(-0.5023940423, 0.5151748053, -0.4771697725, 0.5044840478)`
- Calibration source:
  `/home/lp531/ros2_ws/z1_handeye_charuco_gripper_to_camera.yaml`
- RGB resolution: `640 x 360` (16:9)
- RGB intrinsics, read from the connected D435i's 640x360 factory profile:
  `fx=451.606170654297`, `fy=451.370849609375`,
  `cx=325.197723388672`, `cy=188.640960693359`
- Depth resolution: `640 x 480`
- Depth intrinsics:
  `fx=fy=387.84747314453125`,
  `cx=319.447265625`, `cy=238.202484130859`
- Isaac publishes its renderer-derived values on `camera_info_raw`; the ROS
  launch runs `camera_info_fix.py` to publish the measured matrices on the
  standard RGB/depth `camera_info` topics.
- Clipping range: `0.05–5.0` m
- RGB camera prim:
  `/World/Z1/root_joint/camera_color_optical_frame/ColorCamera`
- Depth camera prim:
  `/World/Z1/root_joint/camera_depth_optical_frame/DepthCamera`

The extra 180-degree local X rotation on each `UsdGeom.Camera` only converts
the ROS optical-frame convention to Isaac's camera convention; it does not
change the physical URDF extrinsic.

The task scene also has a fixed third-person RGB-D observation camera:

- Prim: `/World/GlobalCamera/global_camera_optical_frame/GlobalCamera`
- Position in `world`: `(0.72, -0.72, 0.72)` m
- Look-at point: `(0.32, 0.02, 0.12)` m
- Resolution: `640 x 480`
- Intrinsics: `fx=fy=554.2562584`, `cx=320.0`, `cy=240.0`
- ROS topics: `/global_camera/color/*` and `/global_camera/depth/*`

The global values are a simulation baseline. Replace them with the measured
intrinsics and extrinsics of the real fixed camera before sim-to-real use.

## Open the scene

```bash
/home/lp531/isaac-sim/isaac-sim.sh \
  /home/lp531/isaac_assets/z1/generated/smolvla_red_ball_into_box.usd
```

## Rebuild after changing the Xacro

```bash
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash

xacro \
  /home/lp531/ros2_ws/src/z1_ros2/z1_description/urdf/z1.urdf.xacro \
  sim_ignition:=false \
  with_gripper:=true \
  -o /home/lp531/isaac_assets/z1/generated/z1_with_d435.urdf

# Rebuild normalized visual wrappers after replacing a source USDZ.
/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/prepare_scanned_assets.py

/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/build_smolvla_red_ball_scene.py

/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/smoke_test_smolvla_scene.py
```

The build script intentionally preserves fixed joints so the D435 TF frames
remain addressable. It also removes invalid visual references that Isaac Sim
5.1 authors for empty TF-only URDF links.

## Task layout

- Handle-free chai cup: `(0.62, -0.08, 0.0)` m.
- Apple: `(0.44, -0.18, 0.0)` m.
- Peach: `(0.54, -0.16, 0.0)` m.
- Computer mouse: `(0.49, -0.02, 0.0)` m.
- Beverage bottle: `(0.62, 0.08, 0.0)` m.
- Blue box: center `(0.32, 0.14)` m, outer size `0.16` m, wall height
  `0.10` m.
- The box uses five independent static colliders and has an open top.
- `/World/Task/Box/SuccessVolume` defines the geometric success region.
- Every training object has an independent rigid body, collision geometry,
  mass, CCD, and a `smolvla:semanticLabel`.
- The five objects use high-quality textured scan/render assets. Source
  physics APIs are suppressed in the wrapper layers, and lightweight sphere,
  cylinder, or box colliders are authored by the scene builder for stable,
  faster training.
- Every object has a planned pinch width of `48–52 mm`. Asset and scene
  generation reject objects above the conservative `55 mm` Z1 design limit.

Suggested instructions:

- `Put the cup into the blue box.`
- `Put the apple into the blue box.`
- `Put the peach into the blue box.`
- `Put the computer mouse into the blue box.`
- `Put the beverage bottle into the blue box.`
