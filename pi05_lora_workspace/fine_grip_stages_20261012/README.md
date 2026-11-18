# 闭爪过程细分采样（2026-10-12）

已为全部 211 条示范生成逐帧阶段标签，并将活动训练输入配置切换到本方案。203 条训练、8 条验证；允许训练起点仍为 42,294 个。未启动 LoRA 训练。

## 阶段与权重

| 标记 | 含义 | 采样起点相对权重 |
|---|---|---:|
| other | 其余保留内容 | 1.0 |
| preclose | 开始闭爪前 1 秒 | 1.5 |
| closing | 开始闭爪到末段稳定指令区间 | 2.0 |
| prelift_hold | 抬升前的末段稳定指令区间 | 2.0 |
| initial_lift | 首次 PHYSICAL_TEST_LIFT 到连续 5 帧确认物理抓取且离桌 | 1.5 |
| stable_hold | 确认稳定后的最多 1.5 秒，截止于释放或片段末尾 | 1.0 |

稳定指令区间要求抬升前至少连续 3 帧（10 Hz 下 0.3 秒），从前一帧到抬升前的全部夹爪指令极差不超过 0.005 rad。采用末段稳定区间，避免将中途短暂停顿误标为持握。该标签描述指令状态，不代表已经接触或夹牢。

episode 190（旧纠偏 near_002）没有满足该条件的区间，hold_start=null、hold_frames=0，闭爪阶段一直延续到抬升起点。没有虚构持握帧。

夹爪动作目标仍沿用原有二值标签，机械臂动作、状态、图像和损失函数不变。细分标签用于采样，不是新的模型输入或分类头。

## 纠偏权重恢复与验证

当前方案 fine_grip_with_recovery_20261012：保留上表五阶段权重，并将纠偏记录中的 lift_clear、correct_orientation、align_above、approach 区间（仅闭爪前）赋予至少 2 倍权重。重叠处取 max，不相乘。所有有效权重均不超过 2。普通示范不虚构纠偏区间。

原始/纠偏总概率仍为 70%/30%。相比恢复前的细分方案，实际纠偏动作的采样起点概率从 6.95% 提高到 11.20%，覆盖 1484 个允许起点。这是全部训练抽样中的概率，不是纠偏数据池占比。

本次恢复实际记录的纠偏区间加权；闭爪前仍采用已批准的 1 秒、1.5 倍细分设置，没有整体退回旧 80k 的全部参数。

采样器对 42,294 次抽样进行了区间和验证集隔离检查，并验证确定性恢复。有效采样区间、数据划分、二值标签和归一化保持一致。完整 15 步动作块各阶段覆盖统计见 sampling_audit.json；previous 字段对照的是本次恢复前的细分方案。

before_recovery_restore 保存恢复前的采样数组、配置、审计与准备脚本。6 项阶段/重叠权重测试通过。未启动训练。

## 加载与文件

- 当前训练输入：本目录 `training_input.json`。
- 同步活动配置：`../correction_windows_20261012/training_input.json`。
- 数据集内活动配置：`/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012/meta/training_sampling_config.json`。
- 配置明确指定 `meta/fine_grip_sampling_arrays.npz`；使用 WindowSampler 加载该路径才会生效。
- `phase_boundaries.csv`：211 条的各阶段起止帧和原始记录路径。
- `phase_manifest.json`：阶段边界、原始内容哈希及帧数统计。
- `fine_sampling_arrays.npz`：训练/验证掩码与阶段 ID/权重。
- `previous_training_input.json`：切换前的输入配置。

旧的 merged_211_sampling_arrays.npz 和 meta/sampling_eligibility.npz 保留作粗分阶段对照；当前活动配置已指向新的细分权重文件。再次执行旧 map_dataset.py 会生成粗分阶段配置，完整重建流程最后应执行本目录 prepare.py 重新应用细分。

```sh
/home/lp531/pi05_lora_workspace/.venv/bin/python /home/lp531/pi05_lora_workspace/fine_grip_stages_20261012/prepare.py
/home/lp531/pi05_lora_workspace/.venv/bin/python -m unittest discover -s /home/lp531/pi05_lora_workspace/fine_grip_stages_20261012 -p 'test_*.py'
```
