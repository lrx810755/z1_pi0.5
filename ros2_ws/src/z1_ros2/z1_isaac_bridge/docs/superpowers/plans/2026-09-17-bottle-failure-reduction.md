# Bottle Failure Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove the four dominant bottle collection failure paths found in the first 500 attempts while retaining physical contact, table-collision, and in-box stability checks.

**Architecture:** Keep the existing one-pass bottle grasp and transport pipeline. Make bottle-specific parameters constrain transport dynamics, add a symmetric wrist-roll IK branch to candidate generation, treat 135 mm as the configured accepted box-clearance height while continuing to command 140–150 mm goals, and allow stable physical placement to succeed when only the post-release arm retreat fails.

**Tech Stack:** Python 3.10, ROS 2 Humble, MoveIt, pytest, JSON configuration.

**Spec:** User-approved fixes 1–4 from the 2026-09-17 500-attempt failure analysis.

## Global Constraints

- Preserve the bottle's 140 mm box-approach command and 150 mm preferred transport target.
- Accept bottle clearance at 135 mm, using the configured 5 mm completion tolerance.
- Never bypass real bilateral-contact, table-contact, release, or in-box stability checks.
- Keep a single goal-directed bottle transport; do not restore the old three-stage fallback.
- Preserve all existing dataset episodes and unrelated uncommitted workspace changes.

---

### Task 1: Stop redundant bottle height correction

**Files:**
- Modify: `z1_isaac_bridge/grasp_assistant.py`
- Test: `test/test_grasp_physics.py`

**Interfaces:**
- Consumes: `physical_test_lift_safe_completion_z_m=0.140` and `physical_test_lift_completion_tolerance_m=0.005`.
- Produces: a 0.135 m minimum accepted height passed to loaded transport and used as the lift-loop exit condition.

- [x] Add a failing regression test showing that a bottle already at 0.1397 m does not request another vertical correction.
- [x] Run the focused test and confirm it fails because the lift loop still chases 0.150 m.
- [x] Compute the minimum accepted height from safe completion minus completion tolerance and stop correction once it is reached.
- [x] Run the focused test and existing physical-lift tests.

### Task 2: Bound bottle transport dynamics

**Files:**
- Modify: `config/grasp_physics.json`
- Modify: `z1_isaac_bridge/grasp_assistant.py`
- Test: `test/test_cylinder_transport_clearance.py`
- Test: `test/test_grasp_physics.py`

**Interfaces:**
- Consumes: target-specific bottle profile values.
- Produces: `loaded_transport_orientation_tolerances_rad=(0.20, 0.40)`, `loaded_transport_max_joint_delta_rad=1.70`, and `loaded_transport_acceleration_scale=0.04` while retaining velocity scale 0.08.

- [x] Add failing tests asserting target-specific orientation, joint-delta, velocity, and acceleration options reach the actual move request.
- [x] Run focused tests and confirm current code uses shared orientation/joint limits and one scale for velocity and acceleration.
- [x] Add an optional acceleration scale through `_plan_candidate`, `_execute_manual_pose`, and `_move_object_to`.
- [x] Read bottle-specific transport limits in `_carry_loaded_object_toward_box` and apply them to direct and fallback moves.
- [x] Run focused transport and planning tests.

### Task 3: Add a mirrored bottle wrist IK branch

**Files:**
- Modify: `config/grasp_physics.json`
- Modify: `z1_isaac_bridge/grasp_assistant.py`
- Test: `test/test_grasp_physics.py`

**Interfaces:**
- Produces: `_bottle_candidate_pose_variant(candidate_index, candidate_count, pitch_max_deg, pitch_min_deg, base_roll_deg, mirror_wrist_roll)` returning `(pitch_deg, roll_deg)`.
- Consumes: eight bottle candidates, allocating four pitch values to the taught +90° roll and repeating them with the equivalent -90° roll.

- [x] Add a failing unit test for the four shallow pitches on each of the +90° and -90° wrist branches.
- [x] Run it and confirm the helper/behavior is absent.
- [x] Implement the deterministic variant helper and use it during bottle candidate orientation generation.
- [x] Keep the calibrated object-in-stator transform for both symmetric branches.
- [x] Run candidate geometry and grasp-physics tests.

### Task 4: Preserve stable placement after retreat-only failure

**Files:**
- Modify: `z1_isaac_bridge/grasp_assistant.py`
- Test: `test/test_cylinder_place_motion.py`

**Interfaces:**
- Consumes: `_open_gripper_for_box_release` physical release proof and evaluator `task_success`.
- Produces: successful guided placement when the bottle is released and stable in the box even if the subsequent 40 mm arm retreat raises an error.

- [x] Add a failing behavioral test whose release is valid, retreat raises, and evaluator reports stable task success.
- [x] Add a separate failing test proving retreat failure still fails when placement never becomes stable.
- [x] Run both tests and confirm the existing unconditional retreat exception aborts them.
- [x] Capture the retreat error, publish release completion after the verified open, probe final physics, and let stable `task_success` win.
- [x] Re-raise a useful retreat-plus-placement error when stability never arrives.
- [x] Run all place-motion tests.

### Task 5: Verify and deploy

**Files:**
- Verify all modified source, configuration, and tests.

**Interfaces:**
- Produces: rebuilt ROS 2 package and one restarted bottle-training process.

- [x] Run the complete pytest suite with plugin autoload disabled.
- [x] Run JSON parsing, Python compilation, and `git diff --check`.
- [x] Build `z1_isaac_bridge` using the isolated system Python environment.
- [x] Restart `z1_isaac.launch.py`, wait for all services to report ready, and resume the remaining bottle target count without deleting saved episodes.
- [x] Verify exactly one continuous-training process is active.
