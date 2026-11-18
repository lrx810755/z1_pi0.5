# Z1 gripper compatibility report

Updated: 2026-07-20

## Gripper measurements

Measurements come from the current Z1 URDF and its collision meshes:

- Gripper joint: `jointGripper`
- Joint range: `-90°` to `0°`
- Fixed gripper collision envelope: approximately `156 × 82 × 68.85 mm`
- Scene design limit for a regular pinch region: `55 mm`

The `55 mm` value is intentionally conservative. The Z1 uses a single-DOF
angular jaw, so usable aperture varies with contact depth. Objects with
handles, deep concavities, or a required pinch width above this value are not
accepted by `prepare_scanned_assets.py`.

## Selected objects

| Object | Normalized visual size (mm) | Planned pinch width | Clearance below 55 mm | Recommended grasp |
| --- | ---: | ---: | ---: | --- |
| Chai cup | `50.3 × 50.3 × 61.0` | `50.4 mm` | `4.6 mm` | Side pinch at `35 mm` height |
| Apple | `52.2 × 48.9 × 60.3` | `50.0 mm` | `5.0 mm` | Equatorial pinch |
| Peach | `48.4 × 52.1 × 48.6` | `52.0 mm` | `3.0 mm` | Equatorial pinch |
| Mouse | `88.0 × 49.4 × 28.4` | `48.0 mm` | `7.0 mm` | Pinch across short axis |
| Beverage bottle | `48.1 × 48.3 × 157.0` | `48.4 mm` | `6.6 mm` | Side pinch on body at `75 mm` |

## Rejected cup

The previous coffee mug measured approximately `116.8 × 78.8 × 93.1 mm`;
its physics cylinder was `86 mm` in diameter and its handle increased the
overall width to about `117 mm`. It was replaced by a handle-free chai cup.
The rejected source file is retained under:

`scanned_objects/retired/cup_mug_ungraspable.usdz`

## Validation

- Every selected pinch width is checked during asset-wrapper generation.
- Scene generation repeats the same `55 mm` compatibility check.
- High-detail source physics remains suppressed; the scene uses five
  lightweight colliders.
- Final Isaac Sim runtime smoke test passed without nested-rigid-body or
  high-poly collision-cooking errors.

This report verifies geometric compatibility. Motion-planning reachability
and force-closure should still be verified when implementing each grasp pose.
