# Cup correction100 implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. Runtime collection lives in a new task-local directory, outside production source.

**Goal:** Start a resumable collection of 100 new, complete cup recovery demonstrations using the seven previously saved categories.

**Architecture:** Reuse production ROS/Isaac nodes with task-local subclasses, the previous full-correction geometry/preflight helpers, a deterministic quota plan and independent recorder acceptance. Keep production source and the completed400 dataset unchanged. Start a background supervisor after a physical smoke run and verify the first accepted episode.

**Tech stack:** ROS2 Humble, Isaac Sim RTX, MoveIt, Python3, NumPy/SciPy, raw three-camera episode format.

**Spec:** `/home/lp531/docs/杯子纠偏数据集要求_20261026.md`; user now requests 100 new correction episodes and authorizes recording.

## Global constraints

- 100 new accepted tasks: approach20, close_transition15, shifted_target20, low_clearance10, regrasp10, transport10, placement15; 20 original cells, five tasks per cell.
- All episodes end in verified physical grasp, box placement, release and withdrawal; retain >=16 frames after release; actual FK roll -90±3 degrees before definitive close.
- Three RGB cameras at10Hz, worldEps0.0001 after render initialization, dynamic cup and collisions enabled during motion/recording.
- Synthetic perturbations and setup relocation must be labeled explicitly, with failure-report provenance; do not call them exact policy-state replay.
- When a late-stage disturbance is created during an otherwise continuous full episode, retain the raw frames, mark intervention intervals, and exclude action windows intersecting those intervals from training. Never silently join discontinuous recordings.
- Split cells(1,1),(2,3) as validation. Failures do not count; accepted task IDs unique; per-category quotas fixed.

## Review focus

- Partial retry closure must not be mislabeled as the final release; repeated opening/closing is supported by command-time events.
- Deliberate disturbance actions must not become supervised recovery targets.
- A category label requires actual measured evidence, not only a requested offset.
- Scene relocation cannot occur after recording starts and cannot cross the assigned cell.
- Launching/resuming cannot duplicate an active simulator or count rejected/old episodes.

## Tasks

- [x]1. Test then implement pure quota, labels, training-window mask and category-acceptance protocol in task-local `collector/protocol.py`, tested by `collector/test_protocol.py`.
- [x]2. Implement task-local assistant/GUI/recorder classes, shift scene adapter and isolated launch. Reuse full physical production grasp/place; preplan preparation/recovery, capture pose/contact evidence, reject missing recovery and invalid physical status.
- [x]3. Implement resumable supervisor and launcher: snapshot code/config, count accepted unique tasks, record attempts, cycle missing tasks, stop on repeated no-progress, never relax acceptance.
- [x]4. Run protocol tests and ROS launch/config checks; start isolated scene, review first physical recording and raw images/labels. Obtain a fresh code review, fix material findings, leave collection running with verified progress and user-facing instructions.

## Execution rulings

- User asks to execute the previously saved requirements now; routine quota allocation and task-local implementation proceed without another approval round.
- No production branch edits are needed: the collector is isolated under `z1_queue_logs/cup_correction100_full_renderfix_20261027`.
- Start and first-episode verification fulfill the immediate request; the background supervisor continues toward all100. Never report all100 complete before actual acceptance.
