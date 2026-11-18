# Four Object Evaluation Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Prepare manually launched four-object final-model simulation evaluation.
**Architecture:** Isolated snapshots of the existing ROS rollout and process bridge, a stdlib checkpoint/CLI contract, and a ROS runtime subclass using frozen training provenance. Do not start evaluation during preparation.
**Tech Stack:** Python, Bash, ROS Humble, existing LeRobot/PEFT and Isaac.
**Spec:** ../specs/2026-11-17-four-object-eval-design.md

## Global Constraints
- 800000 microsteps / 100000 optimizer updates only; no intermediate fallback.
- 10Hz, 30 predictions, 15 execution steps, 60Hz interpolation, three RGB inputs, seven joint values, binary gripper.
- Do not modify training inputs, config, output or existing shared runtime.
- No GPU or scene launch during preparation.

## Review Focus
- Incomplete checkpoint must block before launching any process.
- Source/processor changes must not silently pass.
- Bottle aliases and all four target selections must resolve correctly.
- Fixed training seeds must be rejected.
- Default/check/dry-run must not start Isaac or a model worker.

### Task 1: Final model and evaluation contracts
Files: contracts.py, test_contracts.py, source_snapshot.json, grasp_physics.json.
Produces: check_final(run, output, require_reload=False), training_coverage(manifest, physics), parse_args(argv), rollout_arguments(args, checkpoint).
- [x] Write tests for incomplete/final markers, config schema, unsafe seeds and argument modes; run to observe failures.
- [x] Implement stdlib contracts and source provenance snapshots.
- [x] Run contract tests.

### Task 2: Runtime and manual entries
Files: runtime_rollout.py (snapshot), policy_bridge.py (snapshot), entry.py, scene.launch.py, cli.py; SH/pi05_eval.sh, SH/pi05_eval_scene.sh, SH/pi05_verify_final.sh.
Consumes: Task 1 final checkpoint and coverage contract.
- [x] Add real parser/bridge transport/coverage tests, observe failures.
- [x] Implement isolated runtime integration, strict GPU bridge, final reload and scene entry.
- [x] Run CPU integration and shell syntax checks; confirm default/check/dry-run behavior.

### Task 3: Delivery and evidence
Files: verify_cpu.py, README.md, verification.json, progress.md.
- [x] Run actual saved preprocessing on raw images for all eight object/source pools on CPU.
- [x] Run full isolated suite plus relevant existing bridge/runtime regressions.
- [x] Record exact final-model pending status and GPU limitation, commands and output locations.
- [x] Review implementation against design and fix material problems; deliver concise Chinese instructions.
