# Z1 ACT conversion and training handoff — 2026-09-23

## Status and scope

- Interfaces are prepared; **no ACT dataset conversion or training has been run**.
- The balanced raw collection at `/home/lp531/smolvla_data/z1_tabletop_raw_balanced_v2` is still growing. A read-only snapshot during this work found 203 successful episodes: apple 40, peach 42, cup 57, beverage_bottle 64. Recount after collection ends.
- The local LeRobot environment is `/home/lp531/miniconda3/envs/unifolm-vla` (LeRobot 0.3.4). The dataset records 10 Hz, 7-dimensional state/action, and three RGB cameras.
- ACT does not consume the episode's task text. Export and train one target object per model. The first baseline uses `wrist` only, avoiding mixed camera sizes and reducing GPU memory use. Multi-camera ACT and ACT-in-Isaac evaluation are **not** implemented here.

## After collection finishes

Stop the collector first. Confirm that no episode is being recorded and review the saved successes; never convert a moving source or include failed episodes. The following paths are new output directories and must not exist already. Do not delete an older dataset to reuse a name.

Trim redundant leading frames and the duplicate `vla_wrist` stream, preserving the raw source:

```bash
/usr/bin/python3 \
  /home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/prepare_smolvla_dataset.py \
  --source /home/lp531/smolvla_data/z1_tabletop_raw_balanced_v2 \
  --output /home/lp531/smolvla_data/z1_tabletop_act_prepared_v1
```

Convert each target into a separate, success-only, single-camera LeRobot dataset:

```bash
for act_target in apple peach cup beverage_bottle; do
  /home/lp531/miniconda3/envs/unifolm-vla/bin/python \
    /home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py \
    --input /home/lp531/smolvla_data/z1_tabletop_act_prepared_v1 \
    --output "/home/lp531/smolvla_data/z1_tabletop_lerobot_act_${act_target}_v1" \
    --repo-id "lp531/z1_act_${act_target}_v1" \
    --target-object "$act_target" --camera wrist
done
```

The exporter refuses an existing output directory and verifies contiguous frame indices, 7-dimensional state/action arrays, and frame counts. After conversion, inspect each `meta/info.json` and `meta/episodes.jsonl`: one camera key (`observation.images.wrist`), one target task, nonzero episodes, 10 fps, and matching 7-dimensional state/action. Spot-check video, action timing, gripper opening, and final box placement before spending GPU time.

Preview then start one target's ACT trial; logs appear in the terminal and in `/home/lp531/smolvla_training/act_<target>_<timestamp>.log`:

```bash
bash /home/lp531/ros2_ws/src/z1_ros2/scripts/train_z1_act.sh apple --dry-run
bash /home/lp531/ros2_ws/src/z1_ros2/scripts/train_z1_act.sh apple
```

The launcher defaults to batch size 2, 10,000 steps, action chunk 30, 10 executed actions, CUDA, and checkpoints every 2,000 steps. Adjust only if needed with `ACT_BATCH_SIZE` or `ACT_TRAIN_STEPS`; `ACT_DATASET_BASE` and `ACT_TRAINING_ROOT` override the two parent directories. Use a fresh timestamped output directory for each run. Do not run ACT training concurrently with Isaac collection on the same GPU.

## Validation and remaining work

- On 2026-09-23, the new interface tests passed (3/3). The package suite reported 312 passed and 10 failed; those failures concern existing cylinder transport expectations, apple pitch expectations, an absolute launch path, and cup height-transfer expectations. The exporter CLI help, Python syntax, and training-script shell syntax were also checked. No real ACT conversion or training was run.
- Before declaring the model useful, reserve held-out scene layouts and measure actual Isaac pick-and-place success, not just training loss.
- `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/smolvla_sim_rollout.py` loads `SmolVLAPolicy` specifically. ACT inference needs a separate adapter or a reviewed extension before scene evaluation. That is outside this preparation task.
- If a target has zero eligible episodes, the exporter stops with an error; investigate collection coverage rather than training on another target's data.
