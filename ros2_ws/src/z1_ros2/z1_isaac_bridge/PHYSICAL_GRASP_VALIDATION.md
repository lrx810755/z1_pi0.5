# Physical grasp verification

## State machine

```text
freeze target during long planning and pre-grasp motion
-> reach final pre-grasp
-> unfreeze and wait for stable pose/velocity
-> close slowly and detect an actual-angle plateau
-> lift once in a straight line to the 0.14 m transport height
-> verify the final dynamic, gravity, table-clearance and bilateral-contact state
-> latch physical_grasp_ok
-> transport to the blue box
-> stop, hold closed, and open slowly
-> wait for stable natural placement in the box
-> save only when every stage succeeded
```

There is no object-following command or transport constraint in the active
workflow. After `unfreeze`, the target remains a dynamic rigid body for the
rest of the episode.

## Isaac Sim verification

1. Build and launch normally:

   ```bash
   cd /home/lp531/ros2_ws
   source /opt/ros/humble/setup.bash
   colcon build --packages-select z1_isaac_bridge --symlink-install
   source install/setup.bash
   ros2 launch z1_isaac_bridge z1_isaac.launch.py
   ```

2. In another terminal, inspect the staged state:

   ```bash
   source /opt/ros/humble/setup.bash
   source /home/lp531/ros2_ws/install/setup.bash
   ros2 topic echo /smolvla/grasp_control_state
   ros2 topic echo /smolvla/scene_grasp_status
   ros2 topic echo /smolvla/physical_grasp_ok
   ```

3. Open `http://127.0.0.1:8767`, select `apple`, and run
   `一键自动示教`.

4. During `PHYSICAL_TEST_LIFT`, verify that one continuous LIN lift is issued,
   `physical_test_lift_active` is true, physics and gravity are enabled, and
   both fingers report the target.

5. After a successful physical test, expect `physical_grasp_ok=true`, then
   confirm bilateral contact remains valid during each transport segment.

6. At placement, expect a slow-open action and stable in-box verification.
   The recorder saves only after the release/stability predicates are true.

## Current limitations

- The scene status schema contains left/right target-contact objects,
  impulses, consecutive-frame count, table contact, and availability fields.
  PhysX contact reports are now authored on both imported finger links and all
  five target rigid bodies. `bilateral_contact` becomes true only when both
  fingers report the same target for 10 consecutive physics frames. Target
  contact with `/World/Task/WorkSurface` is used directly; the height
  clearance is retained only as a compatibility fallback when report
  subscription fails. The close obstruction signal is still a multi-sample
  actual-joint plateau approximation rather than force control.
- The high-resolution fruit meshes remain visual-only. At scene startup the
  script disables imported Apple/Peach colliders and authors configurable
  slightly inset invisible sphere proxies (`23 mm` and `24 mm` radius) beneath the dynamic
  rigid-body roots. A future asset pass may replace these with measured
  low-poly convex hulls or ellipsoids.
- Before those sphere proxies were added, the scanned Apple collider did not
  fit the open-finger pre-grasp cleanly. The strict envelope check correctly
  rejected that run and discarded the episode.
- With the inset proxy enabled, the final 2026-07-23 run still relaxed outside
  the strict envelope before the collision decomposition fix. After changing
  the imported finger collision wrappers from a single sealed `convexHull` to
  `convexDecomposition`, the target remains in the open-jaw transition region.
- The current approach uses an 86 degree fruit pose, a closed-loop 1 mm
  physical stator-seating step, reduced drive gains, capped effort, and a
  bounded 0.05 rad contact preload.
- The highest-priority remaining item is replacing the spherical fruit proxy
  with a measured ellipsoid or small convex decomposition, then tuning the
  close/seating behavior from recorded left/right contact impulses. A
  controller exposing direct limited force/torque would also make the contact
  preload less dependent on position-drive tuning.
