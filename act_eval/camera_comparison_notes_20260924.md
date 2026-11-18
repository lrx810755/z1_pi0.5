# ACT 瓶子任务相机对照实验记录（2026-09-24 至 2026-09-25）

## 当前结论与状态

**尚不能判断腕部相机是否导致 ACT 效果不佳。** 双相机、仅腕部和仅全局三组模型已经具备；相同固定种子的闭环成功率对照尚未完成。以下固定种子冒烟测试全部在 Isaac 场景验收阶段失败，策略没有执行，不能作为模型失败或成功率样本。2026-09-25 核查时，平衡采集进程仍在运行；本次仅记录现有证据，没有启动新的训练或评估。

## Evidence available now

- The checkpoint at `/home/lp531/pretrained_model` was trained on 100 successful bottle episodes, 40,183 frames, with both `observation.images.wrist` and `observation.images.global`. Its training config specifies 10,000 steps, batch size 2, no image augmentation, and no in-training evaluation (`eval_freq=20000`).
- Three completed one-episode Isaac reports show 0/3 successes: two `grasp_failed`, one `joint_tracking_fault`. These are too few and have mixed failure modes, so they do not identify a camera as the cause. One `grasp_failed` result has a physically implausible final bottle position, which must be excluded from any clean camera comparison.
- Initial wrist frames sampled from training episodes 0, 20, 40, 60, and 80 show the bottle fully visible in some episodes and cut off at the right edge in others. A failed evaluation video also shows the bottle partially outside the wrist view. This establishes a framing risk, not a measured success-rate effect.
- On 20 training frames at frame 80, replacing one camera image with flat gray changed the first predicted action by mean absolute 0.187 rad for wrist and 0.500 rad for global. Replacing that image with an image from the next training episode changed it by 0.266 rad for wrist and 0.106 rad for global. These interventions disagree on which view matters more; both are outside the training distribution and cannot substitute for independently trained camera models.
- On the 20 held-out raw episodes listed in the seed file, the dual-camera checkpoint's first-action mean absolute error at expert frame 80 was 0.0688 rad, similar to 0.0689 rad on 20 sampled training episodes. Swapping wrist or global images with the next held-out episode changed the first predicted action by 0.303 and 0.116 rad respectively. This shows the model reacts to wrist images in some scenes, but open-loop prediction from expert states cannot establish closed-loop success.
- The host has one RTX 4090. After the user authorized pausing collection, its process group and Isaac scene were stopped. The raw dataset then contained 502 episode directories, all with completion summaries. Collection subsequently restarted. The wrist-only and global-only ACT runs each completed 10,000 steps; final checkpoints are `/home/lp531/smolvla_training/act_bottle_wrist_compare_v1/checkpoints/last/pretrained_model` and `/home/lp531/smolvla_training/act_bottle_global_compare_v1/checkpoints/last/pretrained_model`.
- Both single-camera datasets have now been exported from the stable reviewed source. Each contains 100 episodes, 40,183 frames, and the same 7-dimensional state and action arrays as the dual-camera dataset in every episode. The 100 wrist videos and 100 global videos are byte-identical to their counterparts in the dual-camera dataset. Only the unused camera feature was removed.

## Controlled comparison prepared

Train separate ACT models on the same 100 bottle demonstrations with only wrist or only global images; use the existing dual-camera checkpoint as the third model. Keep state/action fields, action chunk, batch size, steps, and scene evaluation settings equal. Run all three on the 20 seeds in `camera_comparison_seeds_20260924.txt` and report physical success, failure category, contact, and whether the bottle is visible in the wrist frame at the start and before grasp.

The 20 seeds come from successful full-randomization bottle episodes in `z1_tabletop_raw_dual_wrist` that were excluded from `z1_tabletop_reviewed_4x100_v1`. They are distinct from the 100 training seeds and all recorded bottle XY positions fall within the training data's observed rectangle `[0.471597, 0.686651] × [-0.266489, -0.007769]` m. The evaluation code now supports `--eval-seeds` and single-camera checkpoints. Both single-camera model training runs completed; fixed-seed runtime replay has not passed scene validation.

## 固定种子仿真测试（2026-09-24）

| 输出目录 | 种子数 | 场景验收 | 策略执行 |
| --- | ---: | --- | --- |
| `camera_compare_smoke_dual` | 1 | 1 次 `episode_setup_failed` | 无 |
| `camera_compare_smoke_three` | 3 | 3 次 `episode_setup_failed` | 无 |
| `camera_compare_smoke_delayed` | 1（重复第一个种子） | 1 次 `episode_setup_failed` | 无 |

前四次使用默认约 0.9 秒的最短稳定等待；重复测试将固定种子等待设为 2 秒。第一个种子 `553892207` 的苹果与桃子角速度分别约为 0.271 与 0.289 rad/s，超过原验收阈值 0.200 rad/s；延长等待后数值几乎未变化。位置漂移仅为微米量级。另三个种子也因苹果和桃子的类似角速度被拒。详情见上述目录的 `report.json` 和 `episodes/*/result.json`。报告中的 0% 是**场景准备失败率对应的占位成功率**，不能解释为 ACT 抓取成功率。

随后在源码中加入了仅供固定种子评估命令使用的参数：等待 2 秒，并将非目标物体角速度上限设为 0.35 rad/s；目标瓶子仍使用 0.20 rad/s，其他位置、接触及穿透检查不变。该调整后的冒烟测试被中断，**没有产生报告，也未验证能否进入策略执行**。这项评估设置与原训练数据采集的场景验收设置不同，正式比较时必须在三组模型上完全一致并单独注明。

更早的三个单回合双相机闭环报告分别为两次 `grasp_failed` 和一次 `joint_tracking_fault`；它们不是本次配对固定种子实验，也不能用于计算三种相机配置的差异。

Treat a camera effect as supported only if one trained view consistently raises physical grasp or complete task success on these paired scenes while controller and simulation faults are counted separately. Review representative videos before attributing a failure to framing.

## 复现实验所需文件与后续步骤

The two exported datasets are at `z1_tabletop_lerobot_act_beverage_bottle_{wrist,global}_compare_v1` below `/home/lp531/smolvla_data`. Both models have already been trained with the settings below; the commands document the runs and are not pending work. The existing dual-camera checkpoint is `/home/lp531/pretrained_model`. The 20 paired seeds are in `/home/lp531/act_eval/camera_comparison_seeds_20260924.txt`.

```bash
for view in wrist global; do
  /home/lp531/miniconda3/envs/unifolm-vla/bin/lerobot-train \
    --dataset.repo_id="lp531/z1_act_beverage_bottle_${view}_compare_v1" \
    --dataset.root="/home/lp531/smolvla_data/z1_tabletop_lerobot_act_beverage_bottle_${view}_compare_v1" \
    --dataset.video_backend=pyav \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=30 --policy.n_action_steps=10 \
    --batch_size=2 --steps=10000 --seed=1000 --num_workers=0 \
    --log_freq=100 --save_freq=2000 --eval_freq=20000 \
    --wandb.enable=false \
    --output_dir="/home/lp531/smolvla_training/act_bottle_${view}_compare_v1"
done
```

After the scene replay can pass validation, run `scripts/run_act_bottle_sim_eval.sh` for each checkpoint with `--checkpoint`, `--episodes-per-object 20`, `--eval-seeds` followed by the integers in the seed file, and a distinct `--output` directory. The wrapper blocks execution while the collector is running. Inspect `report.json`, `result.json`, `trace.jsonl`, and available camera videos after each run. A replay seed rejected by the scene's settling checks must be excluded from all three arms, with the reason recorded. Do not merge setup failures into model success rates.
