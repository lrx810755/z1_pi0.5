# SmolVLA Isaac Sim evaluation

## π0.5 three-camera bottle LoRA (2026-10-06)

Menu option **4**, or `--model pi05`, selects the completed three-camera model:
`/home/lp531/smolvla_training/pi05_bottle_three_camera_20261006/best`.
Best checkpoint: micro step 80000, optimizer update 10000, held-out loss 0.031.
This loss is from the new dataset/split and is not a direct comparison with the old model.

Start the matching scene in one terminal (if it is not already running):

```bash
bash /home/lp531/start_smolvla_bottle_three_camera_scene.sh
```

Then run evaluation in another terminal:

```bash
bash /home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh --model pi05
```

The scene must use `external_camera_layout:=dual`. Inputs are wrist
(`/left_wrist_camera/color/image_raw`), global and global_right. The rollout
subscribes to all three cameras required by the checkpoint; freshness, timestamp
pairing, blank-image checks and video logging include all three. Missing right
camera input prevents policy execution. The launcher selects the same Fast DDS
64 MiB camera profile as the scene.

Training coverage and excluded scene seeds come from the actual 160 raw episodes:
`/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1`.
Results go to `/home/lp531/smolvla_eval/pi05_bottle_three_camera_best`.

Execution remains 10 Hz action points interpolated at 60 Hz, five action points
per prediction, chunk size 50. Bottle freezing is disabled
(`--no-freeze-target-until-close`); the bottle moves freely with collisions enabled.
The bilateral contact wait is disabled (`--prelift-contact-stable-s 0`).
The `pi05` and `pi05-legacy` launch selections enable `--binary-gripper`.
Raw gripper predictions >= -0.75 rad select the demonstrated closed target
0.0 rad; smaller values select the open target -0.90 rad. This adapts DROID's
threshold-to-open/close execution to Z1's existing close-onset criterion, rather
than copying DROID's normalized 0.5 threshold. At 60 Hz each command continues
toward the binary gripper goal subject to the existing per-command limit;
arm interpolation remains active. Published intermediate angles are expected.
Trace rows preserve `policy_action`, `execution_target` and `command`; episode
results save `binary_gripper` and `gripper_binary_threshold_rad`.
Use `--no-binary-gripper` for the continuous-angle baseline, or
`--gripper-binary-threshold-rad VALUE` to tune the threshold. The angle midpoint
between -0.90 and 0.0 is -0.45 rad; it is not the default close-onset threshold.
Closing no longer pauses arm progress to wait for contact. Input pairing defaults remain 25 ms maximum spread,
200 ms maximum age and 500 ms wait timeout.

Menu option **5**, or `--model pi05-legacy`, retains the old two-camera adapter
at `/home/lp531/smolvla_training/pi05_bottle_lora_20261001/best` and its old output
folder. For meaningful comparisons, use each model's matching camera layout.

ROS remains in Python 3.10. PI0.5 runs in the separate Python 3.12 environment
`/home/lp531/pi05_lora_workspace/.venv/bin/python`. The worker loads official base
weights from `/home/lp531/pi05_lora_workspace/weights/pi05_base`, the LoRA adapter,
and saved normalization/tokenizer processors. It returns seven absolute joint
commands. Keep the base weights available.

Interface verification uses saved synchronized RGB/state and issues no robot
commands; it does not measure physical task success. Evidence:
`/home/lp531/smolvla_eval/pi05_three_camera_interface_20261006/`.

## Left wrist two-camera fine-tuning evaluation (2026-09-28)

With the usual Isaac Sim scene running, use the interactive launcher:

```bash
/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh
```

Menu options 1 and 2 select the completed learning-rate-restart models, then
let you choose cup, apple, peach, bottle, or all objects. These checkpoints are
batch 4 step 075605 and batch 8 step 037803. Option 3 selects the new bottle
recovery model (batch 8, step 030000) and defaults to testing the bottle.
The old batch 4 step 151208 checkpoint is no longer a menu option.
For noninteractive use, pass `--model batch4|batch8|bottle-recovery` and
`--objects cup|apple|peach|beverage_bottle`:

```bash
/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh \
  --model batch8 --objects apple
```

To test the bottle recovery model directly:

```bash
/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh \
  --model bottle-recovery
```

This model uses the left wrist and global cameras, the 246-episode raw-format
training coverage snapshot at
`/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1`,
and writes results under
`/home/lp531/smolvla_eval/bottle_recovery_boundary_only_batch8_30000`.
Its final checkpoint is
`/home/lp531/smolvla_training/smolvla_bottle_recovery_boundary_only_batch8_20260930/checkpoints/030000/pretrained_model`.

The batch4 and batch8 choices use the left wrist and global cameras and the
601-episode training coverage snapshot. For their training and failure analysis, see
[SmolVLA训练与验证失败分析_20260928.md](/home/lp531/smolvla_training/SmolVLA训练与验证失败分析_20260928.md).

## Earlier evaluation flow

The rollout node uses the same observation and action schema as the recorded
dataset: three RGB cameras, seven joints in `joint1` through `joint6`,
`jointGripper` order, and the exact per-object training instruction. A policy
chunk contains 50 actions; the node executes the first 10 at 10 Hz, observes
again, and replans.

Keep the normal Z1 Isaac Sim launch running, then start the default 80-episode
evaluation in another terminal:

```bash
/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_sim_eval.sh
```

For one direct policy rollout in the existing Isaac full-randomization scene,
start everything from the normal launch file:

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  run_smolvla_policy:=true target_object:=cup \
  moveit:=false rviz:=false task_gui:=false
```

Change `target_object` to `apple`, `peach`, or `beverage_bottle`. The launch
defaults to one episode and the final 100,000-step checkpoint. Use
`smolvla_episodes:=N` to run more randomized episodes of the selected object.

The default checkpoint is the newest completed checkpoint under
`/home/lp531/smolvla_training/*/checkpoints/last/pretrained_model`. It can be
selected explicitly:

```bash
/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_sim_eval.sh \
  --checkpoint /path/to/checkpoints/last/pretrained_model
```

For a short wiring check before the full run:

```bash
/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_sim_eval.sh \
  --objects cup --episodes-per-object 1 --timeout 60
```

Results are written below `/home/lp531/smolvla_eval`. Every episode contains
the two wrist videos, the global third-person/input video, a command and joint
trace, the final physical predicates, and a categorized result. `report.json`
is updated after every episode and contains overall and per-object success
rates. The first three episodes of each object are marked
`visual_review_required` for video inspection.

During rollout the node stops existing motion, disables the Servo stream, and
publishes `/smolvla/policy_control_active=true`. It rejects non-finite or
wrong-shaped actions, clamps commands to the Z1 joint limits and 0.10 rad per
cycle, and holds position on stale input, finger/table contact, interruption,
or runtime failure.
