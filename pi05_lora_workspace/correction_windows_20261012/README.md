# 全部纠偏数据转换和采样区间映射（2026-10-12）

> 当前活动训练输入已切换到五阶段细分方案：`../fine_grip_stages_20261012/README.md`。training_input.json 指向新的 fine_grip_sampling_arrays.npz；下文粗分阶段权重文件保留用于对照。

已完成旧 32 条、新 19 条纠偏数据的转换与实际索引映射。新 19 条采用用户停止采集时已冻结的合格样本快照，不等待第 20 条。

## 可用于下一次训练的数据

- 合并数据集：`/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012`
- 共 211 条、47,135 帧：原始 160 条 + 旧纠偏 32 条 + 新纠偏 19 条。
- 划分：203 条训练（160 + 28 + 15），8 条验证（旧 4 + 新 4）。
- 新 19 条单独转换结果：`/home/lp531/smolvla_data/z1_bottle_fragments_binary_20261012`，2,490 帧。
- 旧 32 条复用已验证的二值数据；未重复有损编码旧视频。
- 原始 RGB、动作、状态、10 Hz 时序均保留；新片段夹爪标签沿用闭爪意图 0/1 规则，不虚构释放阶段。

## 已应用到采样配置的区间

纠偏开始前最多保留 0.5 秒；没有 recovery stage 的 close_ready 从闭爪前 1 秒开始。连续 5 帧满足 physical_grasp_ok、grasp_success 且不接触桌面后，再保留最多 1.5 秒；终点不达到首次开爪标签、不超过录制末尾。

frame_start/frame_end_inclusive 是连续标签支持区间，anchor_start/anchor_end_inclusive 是允许采样的起点。每个纠偏起点的完整 15 帧动作均在支持区间内。未删除中间帧或重新拼接时间轴。原 160 条普通示范保留原有采样范围。

- 旧 32 条：7,088 帧中区间覆盖 4,413 帧；训练起点 3,462 个。
- 新 19 条：2,490 帧中区间覆盖 1,854 帧；训练起点 1,275 个。
- 合并集允许训练起点共 42,294 个，其中纠偏 4,737 个。
- 验证样本训练掩码全为 false。

## 下次训练的加载约定

读取本目录 `training_input.json`，或数据集内 `meta/training_sampling_config.json`：

1. 数据源使用其中的 dataset_root。
2. 创建采样器时加载本目录 `sampling.py` 的 `WindowSampler`，传入配置中的 sampler_kwargs；其余构造参数沿用 EpisodeAwareSampler。
3. sampler_kwargs.eligibility_path 指向合并数据集内 `meta/sampling_eligibility.npz`，已映射为实际全局行号。
4. 验证集使用配置中的 validation_episodes，并通过 eligible_validation_anchor 选择区间起点。

采样器先过滤区间，再归一化已有 70% 原始 / 30% 纠偏池比例；沿用上一版阶段权重，没有在本任务中另改比例。此比例属于抽样概率，不等于保留帧数比例。

**仅使用 dataset_root 而继续使用普通随机采样器，不会自动生效。** 数据集的标准 info.json 不表达逐帧区间或非连续验证 episode；实际隔离由采样掩码和上述配置完成。本次准备好数据与加载配置，没有启动 LoRA 训练或改写历史训练入口。

## 产物

- `mapped_windows.csv`：51 条纠偏样本的局部帧区间与实际绝对索引。
- `mapped_windows.json`：211 条的实际 episode 索引、来源路径、内容哈希、区间和划分。
- `merged_211_sampling_arrays.npz`：合并集训练/验证起点掩码、阶段权重。
- 数据集内 `meta/correction_sampling_windows.json`、`meta/sampling_eligibility.npz`、`meta/training_sampling_config.json`：与本目录一致的配置副本。
- `mapping_audit.json`：全部 51 条动作和状态、标签、哈希、归一化、实际抽样与恢复验证。
- `media_audit.json`：全部纠偏视频解码及三视角时间边界验证。
- `training_read_audit.json`：通过真实 LeRobotDataset 读取三视角与 15 步动作验证。

`windows.json`、`windows.csv`、`audit.json` 是转换前冻结的区间设计及其历史审计，保留追溯。实际训练使用 mapped_windows.json 和 training_input.json；不要把历史 audit.json 当作当前转换状态。`existing_192_sampling_arrays.npz` 仅用于旧 192 条数据，不可用于 211 条数据。

## 复现

```sh
/home/lp531/pi05_lora_workspace/.venv/bin/python /home/lp531/pi05_lora_workspace/correction_fragments_20261012/prepare.py --accepted-snapshot /home/lp531/pi05_lora_workspace/correction_windows_20261012/windows.json
/home/lp531/pi05_lora_workspace/.venv/bin/python /home/lp531/pi05_lora_workspace/correction_windows_20261012/map_dataset.py
/home/lp531/pi05_lora_workspace/.venv/bin/python /home/lp531/pi05_lora_workspace/correction_windows_20261012/verify_media.py
/home/lp531/pi05_lora_workspace/.venv/bin/python /home/lp531/pi05_lora_workspace/correction_windows_20261012/verify_training_read.py
/home/lp531/pi05_lora_workspace/.venv/bin/python -m unittest discover -s /home/lp531/pi05_lora_workspace/correction_windows_20261012 -p 'test_*.py'
/home/lp531/pi05_lora_workspace/.venv/bin/python -m unittest discover -s /home/lp531/pi05_lora_workspace/correction_fragments_20261012 -p 'test_*.py'
```
