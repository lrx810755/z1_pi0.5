# Z1 Isaac Sim ROS 2 bridge

This package preserves the controller action names already used by `z1_moveit`
and routes them to the Isaac Sim scene without loading Gazebo or the real Z1
hardware plugin.

The trajectory adapter accepts up to `0.002 rad` of numerical drift at a URDF
joint boundary and clamps such points back to the legal limit. This prevents
small PhysX zero-position noise from rejecting otherwise valid MoveIt plans.

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

Useful launch switches:

```bash
# Headless integration check, without MoveIt/RViz
ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  rviz:=false moveit:=false start_isaac:=true

# ROS side only; start Isaac manually in another terminal
ros2 launch z1_isaac_bridge z1_isaac.launch.py start_isaac:=false
```

For a manual Isaac process, set its bundled ROS library path before the process
starts:

```bash
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
  ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  LD_LIBRARY_PATH=/home/lp531/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib \
  /home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py
```

ROS interfaces:

- `/joint_states`
- `/isaac_joint_commands`
- `/joint_trajectory_controller/follow_joint_trajectory`
- `/gripper_trajectory_controller/follow_joint_trajectory`
- `/clock`
- `/camera/camera/color/image_raw`
- `/camera/camera/color/camera_info`
- `/camera/camera/depth/image_rect_raw`
- `/camera/camera/depth/camera_info`
- `/global_camera/color/image_raw`
- `/global_camera/color/camera_info`
- `/global_camera/depth/image_rect_raw`
- `/global_camera/depth/camera_info`

The wrist RGB stream uses the connected D435i's factory-calibrated 16:9
profile: `640 x 360`, `fx=451.6062`, `fy=451.3708`, `cx=325.1977`,
`cy=188.6410`. Wrist depth and both global-camera streams remain `640 x 480`.

## Grasp and placement evaluator

The one-command launch also starts `grasp_evaluator`. Isaac publishes the five
dynamic object poses on `/isaac/object_transforms`; the evaluator combines the
selected object's ground-truth pose with `/joint_states` and the
`world -> gripperStator` TF.

The fixed global RGB-D camera is placed at `[0.72, -0.72, 0.72] m` and looks
at the tabletop center. Its 640 x 480 pinhole model has `fx=fy=554.2563` and a
roughly 60 degree horizontal field of view; its optical center is `(320,240)`.
Both aligned streams use
`global_camera_optical_frame`; the one-command launch publishes its static
`world` transform. These are simulation baseline values and should be replaced
with the measured intrinsics and extrinsics of the real global camera before
sim-to-real training.

Select the initial target at launch:

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py target_object:=apple
```

Or change it while running:

```bash
ros2 topic pub --once /smolvla/target_object std_msgs/msg/String "{data: cup}"
```

Supported names are `cup`, `apple`, `peach`, `mouse`, and
`beverage_bottle`.

Outputs:

- `/smolvla/grasp_success` and `/smolvla/physical_grasp_ok`: the selected
  object passed the loaded physical test lift.
- `/smolvla/task_success`: a grasp was previously confirmed, then the object
  was released inside the box success volume and remained slow for 0.5 s.
- `/smolvla/task_phase`: `WAITING_FOR_STATE`, `READY`, `GRASPING`, `LIFTING`,
  `TRANSPORT`, `PLACING`, or `SUCCESS`.
- `/smolvla/grasped_object`: selected object name while it is held.
- `/smolvla/status`: JSON diagnostics with the individual predicates,
  distances, height, speed, and failure reason.

Reset only the evaluator latch and recapture tabletop heights:

```bash
ros2 service call /smolvla/reset_evaluator std_srvs/srv/Trigger "{}"
```

This is a simulation ground-truth evaluator. It does not claim to reproduce a
real tactile/contact sensor. Grasp confirmation requires an obstructed slow
close, the shared stator-local grasp envelope, at least 25 mm of physical
object lift, bounded relative translation/rotation, table clearance, and
real bilateral contact. The active workflow has no object-following or
transport-constraint feature.

Final grasp alignment uses one shared closed-loop controller for every object.
MoveIt/Pilz supplies collision-checked Cartesian segments, but completion is
decided from the subsequent Isaac stator-local probe rather than from the
planned endpoint alone. Small residuals are corrected proportionally and
bounded without a forced minimum step; axial insertion pauses while lateral or
vertical alignment remains outside the selected object's configured geometry.
Cup, apple, peach, mouse, and beverage-bottle profiles retain separate target
depths and recoverable Y/Z tolerances. The outer, real-collision insertion, and
final pre-close stages all use the same feedback rule.

## Task GUI

The launch file starts `grasp_gui_bridge` by default. After launching the
stack, open:

```text
http://127.0.0.1:8767
```

The page shows ROS connectivity, phase, grasp/task results, object pose,
gripper opening, gripper-to-object distance, target-box state, and failure
reason. Its red `一键重置 Episode` button stops active trajectories, discards
an unfinished recording, restores the arm/gripper and all five objects, clears
the evaluator/pre-grasp state, detaches MoveIt grasp collisions, and reloads
the dynamic collision scene. The same operation is available from ROS:

```bash
ros2 service call /smolvla/reset_episode std_srvs/srv/Trigger "{}"
ros2 topic echo /smolvla/episode_reset_status
```

Reset uses a near-zero safe home (`joint2=+0.002`, `joint3=-0.002 rad`) and an
open gripper, avoiding exact-limit numerical drift while remaining equivalent
to the MoveIt SRDF `home` pose.
The older `仅复位判定器` button remains available for cases where the physical
scene must stay unchanged.

Useful switches:

```bash
# Use another local port
ros2 launch z1_isaac_bridge z1_isaac.launch.py task_gui_port:=8770

# Disable the web bridge
ros2 launch z1_isaac_bridge z1_isaac.launch.py task_gui:=false
```

The legacy teleoperation frontend at port `8765` also proxies this state when
its backend is used. It expects the ROS GUI bridge at
`Z1_SMOLVLA_GUI_BRIDGE_URL` (default `http://127.0.0.1:8767`). Do not start its
legacy Isaac backend alongside the current one-command scene merely to view
the task panel; use port `8767` with RViz teleoperation instead.

## Assisted pre-grasp

The same launch starts `grasp_assistant` when MoveIt and Isaac are enabled.
For normal dataset collection, select an object and press `一键自动示教`. The
backend performs target confirmation, full reset, planning, recording, guided
pre-grasp, grasp verification, placement verification, and save as one locked
transaction. Planning happens before recording. A successful run is saved
automatically; any failure discards the unfinished recording but leaves the
robot and scene at the failure state for inspection. Press
`一键重置 Episode` manually when ready to restore the scene. `/api/status`
exposes progress under `auto_demonstration`.

The four-step workflow remains available for debugging and manual review:

1. `生成并预览预抓取轨迹` samples eight object-specific poses and publishes
   the shortest feasible MoveIt result to `/display_planned_path`.
2. `开始录制并移动到预抓取位` starts the recorder and sends that exact
   trajectory to Isaac.
3. `自动对准、抓取并抬升` reaches the final pre-grasp while the target is
   frozen, restores dynamic physics, waits for stability, closes slowly, and
   performs one continuous Cartesian lift to the 0.14 m transport height.
4. `自动放入蓝盒并释放` preserves the verified grasp orientation during
   MoveIt horizontal transport, lowers to 0.115 m, releases above the box
   walls, and waits for the stable placement evaluator. The transport planner
   retries increasing orientation tolerances but still rejects joint changes
   above 2.4 rad.

Failed runs discard the unfinished recording but remain at the failure pose
until `一键重置 Episode` is pressed manually. PhysX reports are enabled on
both finger links and all five target rigid bodies; bilateral contact must
persist for 10 physics frames and table support comes from the reported
target/work-surface contact pair.

Candidate azimuth, elevation, and standoff vary each time. The link06
pre-grasp origin is kept at or above 0.12 m. The wrist orientation includes a
120 degree tool-axis roll so the calibrated D435 stays above/to the side of the
fingers instead of colliding with the target or table.

For stable expert demonstrations, the guided path freezes only the selected
Isaac rigid body during the long pre-grasp motion. At the final pre-grasp pose
it unfreezes the body before closing. One continuous vertical LIN motion
raises the loaded object directly to the box-wall clearance height; the old
fixed-height segmented test lift and its OMPL sway fallback are not used.
Lift and transport use only gravity, collision and bilateral finger contact.
At the box, the robot stops, opens slowly, and waits for a stable natural
placement. MoveIt collision checking for the target, table, box, camera, and
robot remains active throughout.

ROS interfaces:

- `/smolvla/auto_generate_pregrasp` and
  `/smolvla/auto_execute_pregrasp` (`std_srvs/srv/Trigger`, internal)
- `/smolvla/auto_guided_grasp` and `/smolvla/auto_guided_place`
  (`std_srvs/srv/Trigger`, internal)
- `/smolvla/stop_assistant` (`std_srvs/srv/Trigger`, reset support)
- `/smolvla/pregrasp_status` (`std_msgs/msg/String`, JSON)
- `/smolvla/scene_grasp_command` and `/smolvla/scene_grasp_status`
- `/smolvla/grasp_control_state` (`std_msgs/msg/String`, JSON)
- `/smolvla/physical_grasp_ok` (`std_msgs/msg/Bool`)

The shared physical thresholds and the single stator-local grasp envelope are
stored in `config/grasp_physics.json`. Both ROS 2 nodes and the launched Isaac
scene receive that same file; do not duplicate those dimensions in scripts.
See `PHYSICAL_GRASP_VALIDATION.md` for the staged state machine, verification
commands, and current sensor/collision limitations.

The local GUI endpoint for the complete transaction is
`POST /api/demonstration/auto` with JSON
`{"target_object":"apple"}`. Duplicate requests and starts while another
recording is active are rejected.

Every automatic demonstration now calls the randomized Episode reset before
planning. The web GUI exposes three curriculum levels:

- `easy`: local ±60 mm translation, at least 35 mm actual movement, and
  ±45 degrees yaw.
- `medium`: local ±120 mm translation, at least 60 mm actual movement, and
  ±90 degrees yaw.
- `full`: all object classes share the same safe workspace, move at least
  80 mm from their fixed asset poses, and use ±180 degrees yaw. Sampling order
  is shuffled so object identity is not tied to a fixed region.

Every level rejects layouts without at least 20 mm of inter-object clearance,
keeps objects inside the configured reachable workspace, and excludes the blue
placement box. A sampled layout is no longer accepted immediately after its
poses are written. Isaac first runs the dynamic bodies with gravity, then a
separate validation checks each object's horizontal and vertical settling
drift, linear and angular speed, orientation drift, real table contact, and
maximum PhysX contact penetration. Only the settled poses are published as
`object_positions_m`; the requested poses and per-object settle metrics remain
in the status payload for diagnostics. A failed validation rejects the seed
before MoveIt is refreshed or recording begins.

The returned message contains the level, seed, and observed translation range.
Available interfaces:

Randomization is also target-aware. The selected target is sampled first, then
all distractors are rejected if their collision footprint enters the gripper's
approach envelope, finger-closing area, or initial vertical-lift area. The
approach envelope is asymmetric: it is wider on the robot side to cover the
wrist camera and gripper stator during outer alignment, while the narrow
target-side cap avoids rejecting safe distractors beyond the object. The grasp
assistant independently checks the final candidate against current Isaac object
poses before accepting a MoveIt pre-grasp plan. Automatic demonstrations retry
up to three fresh layouts before recording starts when a layout or pre-grasp
candidate is blocked.

- `POST /api/randomize_episode` or
  `/smolvla/randomize_episode` (`std_srvs/srv/Trigger`) performs the complete
  stop/reset/randomize/state-clear/MoveIt-refresh transaction. The legacy ROS
  service remains an `easy` alias; the HTTP endpoint accepts
  `{"curriculum_level": "easy|medium|full"}`.
- `/smolvla/randomize_episode_easy`,
  `/smolvla/randomize_episode_medium`, and
  `/smolvla/randomize_episode_full` expose explicit ROS curriculum services.
- `/smolvla/randomize_scene` (`std_srvs/srv/Trigger`) randomizes the Isaac
  objects directly using `easy`; corresponding `_easy`, `_medium`, and `_full`
  scene services are also available. Low-level users must allow the configured
  settling duration and then call `/smolvla/validate_randomized_scene`; the
  complete Episode services perform both steps automatically.
- Publish `{"seed": 12345, "level": "medium"}` on
  `/smolvla/scene_randomization_command` (`std_msgs/msg/String`) to reproduce
  an exact layout. This topic path schedules validation automatically after the
  settling duration.

Each recorded Episode copies the selected curriculum level, seed, initial
object positions, displacement measurements, sweep-clearance target, and
clearance result into both `episode_start.json` and `episode_summary.json`.

The MoveIt planning scene includes the six static tabletop/box primitives and
all five dynamic training objects. Robot self-collision remains enabled while
only the selected target is permitted to touch the gripper links. The web GUI
does not expose staged pre-grasp or end-effector fine-adjustment commands;
training collection is driven by the atomic auto-demonstration endpoint.

## MoveIt tabletop collision scene

When MoveIt is enabled, `planning_scene_sync` registers six primitive collision
objects through `/apply_planning_scene`:

- `smolvla_table`: `0.90 x 0.70 x 0.05 m`, centered at
  `[0.30, 0.00, -0.025]`;
- `smolvla_box_bottom`: `0.16 x 0.16 x 0.01 m`;
- `smolvla_box_left`, `right`, `front`, and `back`: the four 0.10 m high box
  walls with 0.01 m thickness.

The coordinates come directly from the Isaac scene generator and use the Z1
`link00` coordinate convention. The GUI reports `已载入 6 个` after MoveIt
accepts the scene. Manually reapply it with:

```bash
ros2 service call /smolvla/reload_collision_scene std_srvs/srv/Trigger "{}"
```

For an authoritative check, request full world geometry instead of reading the
first incremental `/monitored_planning_scene` message:

```bash
ros2 service call /get_planning_scene moveit_msgs/srv/GetPlanningScene \
  "{components: {components: 24}}"
```

## Dynamic object collision synchronization

`object_collision_sync` also registers the five training objects from
`/isaac/object_transforms` using graspable primitive geometry:

- `smolvla_object_cup`, `apple`, `peach`, `mouse`, and `beverage_bottle`;
- poses are refreshed at 5 Hz after a 2 mm or 2 degree change;
- only the selected target may contact `link06`, `gripperStator`, and
  `gripperMover`; collisions with the remaining arm links stay enabled;
- every target always remains a world collision object.

The GUI reports `已同步 5 个`. Inspect the JSON state on
`/smolvla/object_collision_status`, or manually reload the world objects:

```bash
ros2 service call /smolvla/reload_object_collisions std_srvs/srv/Trigger "{}"
```

## SmallVLA episode recording

`episode_recorder` starts with the one-command launch. It samples at 10 Hz and
stores two RGB observations, the seven measured Z1 joints, the seven commanded
joints, language instruction, task phase, and success labels. The default raw
dataset root is `/home/lp531/smolvla_data/z1_tabletop_raw_16x9`. The separate
root prevents the new `640 x 360` wrist observations from being mixed with
earlier `640 x 480` episodes.

Use the recording card in the web GUI, or call the services directly:

```bash
ros2 service call /smolvla/recording/start std_srvs/srv/Trigger "{}"
ros2 service call /smolvla/recording/save std_srvs/srv/Trigger "{}"
ros2 service call /smolvla/recording/discard std_srvs/srv/Trigger "{}"
ros2 topic echo /smolvla/recording_status
```

After a failed demonstration, use `/smolvla/reset_episode`; an active
`.recording_NNNNNN` is discarded automatically before the scene is restored.

Saving requires at least ten frames and a successful task judgment by default.
Each raw episode is first written under `.recording_NNNNNN`, then atomically
renamed to `episode_NNNNNN` on save. An interrupted directory is preserved for
diagnosis rather than silently treated as training data.

Convert all successful raw episodes to the local LeRobot 0.3.4 format with:

```bash
conda run -n unifolm-vla python \
  /home/lp531/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py
```

The output defaults to `/home/lp531/smolvla_data/z1_tabletop_lerobot_16x9` and
contains `observation.images.wrist`, `observation.images.global`,
`observation.state`, `action`, and the natural-language task. Choose a new
`--output` path for another export; the converter never overwrites a dataset.
