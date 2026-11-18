# SmolVLA Voice Control Design

## Goal

Provide a backend-first voice interface for the trained SmolVLA policy. After
the operator types `start`, the system performs one FULL scene randomization,
waits for a spoken grasp command, maps it to one of the four exact training
tasks, and executes the existing protected rollout. Every completed or failed
attempt returns to `IDLE`; another attempt requires another typed `start`.

## Architecture

Two ROS 2 processes preserve the working Python environments. The voice
frontend runs in the `robot` Conda environment and reuses `robot/asr.py`
(Whisper plus local Ollama `qwen2.5:3b`, with rule fallback). It publishes only
structured `start`, `grasp`, and `stop` messages. The policy backend runs in
`unifolm-vla`, loads the new two-camera/state13 checkpoint, and reuses
`SmolVLASimRollout` for observation construction, chunk inference, joint
limits, tracking protection, contact protection, grasp phase signaling, and
result recording.

The backend publishes `IDLE`, `RANDOMIZING`, `LISTENING`, `EXECUTING`, and
terminal result states on a transient-local status topic. FULL randomization
happens only on `start`. A later grasp command selects the target and exact
`TASK_INSTRUCTIONS` string without changing the prepared scene. A stop command
sets a thread-safe event checked inside the rollout loop.

## Intent Contract

Only `cup`, `apple`, `peach`, and `beverage_bottle` may reach the backend. The
frontend accepts common Chinese and English aliases, while the LLM is asked for
a constrained JSON action/object result. Ambiguous or unsupported requests do
not move the robot. The natural-language ASR output is never passed directly
to SmolVLA.

## Operation

One launcher detects whether the Isaac/ROS stack is already available. It
reuses a live scene or starts `z1_isaac.launch.py`, waits for the FULL
randomization service, starts the policy backend, and keeps the voice frontend
in the terminal. The terminal command `start` begins an attempt; `quit` exits.
During execution, spoken or terminal stop commands request a protected halt.

## Verification

Pure tests cover alias mapping, command validation, state transitions, exact
task tokens, prepared-scene execution, and operator stop. Build the ROS package
once and run the launcher only far enough to confirm service/model startup; do
not trigger a physical rollout automatically.
