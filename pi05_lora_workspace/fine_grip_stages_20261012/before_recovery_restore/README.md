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

## 概率含义与验证

按起点阶段赋权后，在原始/纠偏两个池内归一化，保持 70%/30% 的总概率。各阶段不设固定配额；权重不等于最终占比。

按所有可采样起点精确计算：

- 闭爪前：5.96% → 6.72%。
- 正在闭爪：6.31% → 7.11%。
- 保持闭合准备抬升：3.30% → 3.71%。
- 初始抬升：42.67% → 35.99%。

完整 15 步动作块内，有效动作位置的“正在闭爪＋准备抬升”覆盖率从 8.27% → 9.33%。这是实际包含标签的覆盖统计，不是损失贡献估计。

普通示范保留之前的尾部采样方式，因此动作块覆盖统计排除了跨过普通示范末尾的填充位置。所有纠偏起点仍保证完整 15 步、不跨区间边界。没有为改善统计而删改普通示范。

42,294 次实际抽样通过区间和验证集隔离检查；恢复同一随机采样序列检查通过。统计同时包含精确理论概率和实际抽样经验比例，见 sampling_audit.json。

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
