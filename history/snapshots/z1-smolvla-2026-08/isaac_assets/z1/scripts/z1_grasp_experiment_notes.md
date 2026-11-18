# Z1 grasp experiment notes

Timestamp: 2026-06-03 19:11 CST

## Current results

- Camera visual code defaults to `Z1_CAMERA_VISUAL_ENABLED=0`; runtime code creates only a small white `VisibleCameraBody`.
- MoveIt bridge accepts Isaac current arm joint positions as `start_joint_state`, so MoveIt planning no longer starts from a stale state.
- MoveIt pregrasp, grasp, and lift planning all returned `success=true` in the latest runs.
- Isaac trajectory execution now runs; the previous target overwrite issue was removed.
- Red ball is now a `DynamicSphere` physical prim, not a `VisualSphere`.
- USD `FixedJoint` creation marks the red ball as attached, but the runtime PhysX joint alone did not reliably lift the ball.
- Assisted attachment was added as a temporary fallback, but final ball z still did not reliably match lift height because the Isaac `/World/z1_debug/link06` prim frame does not match the MoveIt end-effector pose exactly.

## Problems to fix next

- First priority: remove every red camera obstruction from the wrist camera view.
- Confirm whether the red block is authored inside `z1.usd` under `WristCamera` or a mount child.
- After camera obstruction is verified gone, continue grasp optimization.
- Replace temporary assisted attachment with a cleaner physical grasp implementation, likely using Isaac surface gripper schema or a correctly refreshed runtime joint anchored to the actual gripper contact frame.
- Align Isaac end-effector/contact frame with the MoveIt `link06` frame before relying on link-local attachment.

## Camera obstruction verification

Timestamp: 2026-06-03 19:30 CST

- Applied `purge_wrist_camera_obstructions.py` to `/home/lp531/isaac_assets/z1/z1.usd`.
- The script preserved a USD backup at `/home/lp531/isaac_assets/z1/z1.usd.camera_obstruction_backup`.
- Runtime validation was performed with `Z1_RED_BALL_ENABLED=0` and `Z1_CAMERA_VISUAL_ENABLED=0`.
- Captured `/api/camera.bmp` to `/tmp/z1_camera_after_usd_purge.bmp`.
- Red pixel check result: `red_pixels=0`, `red_ratio=0.0`, image shape `(480, 640, 3)`.
- Visual inspection of `/tmp/z1_camera_after_usd_purge.png` shows no red camera obstruction block.

Next rule: before continuing grasp optimization, keep the red camera obstruction regression check as the first validation step.
