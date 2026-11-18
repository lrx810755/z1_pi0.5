# Z1 bottle grasp handoff

Last updated: 2026-09-04 (Asia/Shanghai)

## Operator-confirmed ground truth

- The bottle localization/final grasp pose is consistently accurate.
- The remaining false failures are in the post-close physical validation, not in pose planning.
- From the detected contact angle, commanding an additional **3 degrees** of gripper closure is sufficient to lift the bottle successfully.

## Current intended behavior

- Bottle approach candidates remain enabled over URDF pitch `-5` to `-15` degrees.
- The selected candidate reaches the final pre-close pose with one MoveIt trajectory.
- MoveIt Servo is disabled by default.
- Pre-close 3D alignment guards remain disabled for the current unprotected simulation trial.
- No extra MoveIt global plan is allowed between final localization and closing.
- Closing still requires real finger/object contact evidence and rejects finger/table contact.
- After contact, the bottle path applies `0.0523598776 rad` (exactly 3 degrees) of finite preload.
- If that first validation fails while real bilateral mesh contact, nonzero forces, the loaded envelope, and table clearance are still safe, exactly one additional `0.0523598776 rad` (3-degree) preload is attempted before revalidation.
- Post-close acceptance requires 2 consecutive valid physical samples. It still requires bilateral target contact, balanced nonzero impulses, valid bottle-body contact height, and the loaded-pose envelope.
- The bottle-body contact-height band is `0.055` to `0.102 m`; the upper limit includes a narrow 2 mm measurement allowance after an otherwise valid grasp was observed at `0.101235 m`.

## Files changed

- `ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json`
  - `object_grasp_profiles.beverage_bottle.single_contact_preload_rad = 0.0523598776`
  - `object_grasp_profiles.beverage_bottle.loaded_samples_required = 2`
  - `object_grasp_profiles.beverage_bottle.grasp_height_range_m = [0.055, 0.102]`
  - `object_grasp_profiles.beverage_bottle.single_contact_retry_preload_rad = 0.0523598776`
  - `object_grasp_profiles.beverage_bottle.single_contact_retry_attempts = 1`
- `ros2_ws/src/z1_ros2/z1_isaac_bridge/test/test_grasp_physics.py`
  - Regression test verifies the exact 3-degree preload and 2-sample criterion.

The same changes are mirrored in the active workspace at `/home/lp531/ros2_ws`.

## Verification completed

```text
93 tests passed
colcon build --packages-select z1_isaac_bridge: success
installed grasp_physics.json contains the new values
```

The ROS 2 launch that was already running before this update still has the old Python process and must be restarted before the new values are active.

## Next validation

Restart the existing Z1 launch, perform one bottle grasp, and inspect the first failure only. Do not re-enable Servo or add another global MoveIt plan. If it still fails, capture:

- detected contact angle and commanded hold target;
- left/right contact objects, contact points, impulses, and consecutive frames;
- bottle local position at post-close validation;
- exact rejection predicate;
- bottle world Z before and after lift.

If the bottle visibly lifts while software rejects it, adjust only the predicate contradicted by that successful physical evidence; do not move the calibrated final grasp pose.
