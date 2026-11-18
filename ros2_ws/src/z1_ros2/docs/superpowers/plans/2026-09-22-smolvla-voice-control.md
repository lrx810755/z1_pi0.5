# SmolVLA Voice Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an interactive voice-controlled SmolVLA rollout that FULL-randomizes on typed `start` and requires another `start` after every attempt.

**Architecture:** A ROS voice frontend in the `robot` environment publishes constrained commands to a policy backend in `unifolm-vla`. The backend reuses the current rollout and splits scene preparation from execution so a spoken target does not cause a second randomization.

**Tech Stack:** Python 3.10, ROS 2 Humble, rclpy, Whisper, Ollama/LangChain, SmolVLA, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-22-smolvla-voice-control-design.md`

## Global Constraints

- Valid targets are exactly `cup`, `apple`, `peach`, and `beverage_bottle`.
- SmolVLA receives only the existing exact `TASK_INSTRUCTIONS` text.
- FULL randomization occurs once per typed `start`, before voice input.
- Every attempt returns to `IDLE` and needs another `start`.
- Existing rollout safety and control behavior remains shared.
- No new Python dependencies.

## Review Focus

- Reject grasp commands received outside `LISTENING` without moving.
- Reject unsupported or ambiguous ASR objects.
- Ensure prepared execution never performs a second randomization.
- Ensure `stop` is observed while policy execution is blocked in another thread.
- Ensure launcher cleanup does not terminate a pre-existing Isaac scene.

---

### Task 1: Voice command core

**Files:**
- Create: `z1_isaac_bridge/z1_isaac_bridge/smolvla_voice_core.py`
- Create: `z1_isaac_bridge/test/test_smolvla_voice_core.py`

**Interfaces:**
- Produces: `match_target(text, parsed_object) -> str | None`, `decode_command(payload) -> VoiceCommand`, and `VoiceSession` transitions.

- [ ] Write tests for aliases, malformed JSON, invalid states, terminal completion, and stop.
- [ ] Run the focused test and verify missing imports fail.
- [ ] Implement the smallest pure core that passes.
- [ ] Run the focused test and existing rollout-core tests.

### Task 2: Prepared rollout and cancellation

**Files:**
- Modify: `z1_isaac_bridge/z1_isaac_bridge/smolvla_sim_rollout.py`
- Modify: `z1_isaac_bridge/test/test_smolvla_rollout_core.py`

**Interfaces:**
- Produces: `_randomize_scene() -> int`, `_activate_policy(target)`, and `run_episode(..., prepared_seed=None, stop_event=None)`.

- [ ] Add source-level and pure behavior tests proving prepared execution skips preparation and stop has a failure category.
- [ ] Run tests and observe the expected failures.
- [ ] Split randomization/activation while preserving evaluation behavior; check cancellation in both rollout loops.
- [ ] Run focused tests.

### Task 3: ROS backend and ASR frontend

**Files:**
- Create: `z1_isaac_bridge/z1_isaac_bridge/smolvla_voice_backend.py`
- Create: `/home/lp531/robot/smolvla_voice_frontend.py`
- Modify: `z1_isaac_bridge/setup.py`
- Modify: `z1_isaac_bridge/test/test_smolvla_voice_core.py`

**Interfaces:**
- Consumes: JSON commands on `/smolvla/voice_command`.
- Produces: transient-local JSON status on `/smolvla/voice_status`.

- [ ] Add tests for backend command gating and exact task lookup.
- [ ] Run tests and observe missing backend behavior.
- [ ] Implement backend queue/status handling and frontend `start`/wake-word loop.
- [ ] Run focused tests and compile both Python files.

### Task 4: One-command launcher

**Files:**
- Create: `/home/lp531/start_smolvla_voice.sh`

**Interfaces:**
- Starts or reuses the Isaac stack, backend, and foreground frontend.

- [ ] Add a shell syntax and static command assertion to the platform launcher test.
- [ ] Run the test and observe failure because the launcher is absent.
- [ ] Implement service detection, startup wait, environment selection, and ownership-aware cleanup.
- [ ] Run shell syntax, focused tests, ROS build, and one startup check without typing `start`.
