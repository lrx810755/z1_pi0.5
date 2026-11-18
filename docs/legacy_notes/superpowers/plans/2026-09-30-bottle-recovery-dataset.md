# 瓶子纠偏训练集重新生成计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从原始瓶子录制重新生成训练集，保留所有中间帧，仅允许删除开头与末尾连续重复帧。

**Architecture:** 单独准备脚本读取四个原始目录，检查数据并确定每条轨迹唯一的连续保留区间，生成新的原始格式目录及映射清单。复用现有 LeRobot 导出工具逐帧转换，最后验证输出与来源一一对应。

**Tech Stack:** Python 标准库、现有 NumPy/OpenCV、unifolm-vla 环境中的 LeRobot。

**Spec:** 用户本轮要求：“生成新的训练集，不允许再裁剪中间帧，只允许裁剪开头与末尾重复帧，在重新生成计划”。本文件记录其执行规则。

## Global Constraints

- 从原始 frames.jsonl 和图像生成，不以 prepared、trimmed 或旧 LeRobot 数据作为输入。
- 不允许裁剪中间帧，包括静止、等待、纠偏、接触、闭爪、搬运、释放等片段；不得按 phase 删除帧、全局去重、跳帧或重采样。
- 每条输出只允许是原始轨迹的连续区间 [start, end)，至少保留一帧。中间出现重复帧也全部保留。
- 重复定义采用保守的完全一致：observation.state、action、两路解码后的 RGB 图像以及 phase、任务/抓取/接触标记相同。时间戳和帧序号不参与重复比较。字段缺失或无法判断时不裁剪，并报告原因。状态或图像仅近似相同不算重复。
- 仅压缩与第一帧完全相同的连续前缀、与最后一帧完全相同的连续后缀；前缀保留其最后一帧，后缀保留其第一帧。整条完全重复时保留一帧并标为待复核，不自动进入最终训练集。
- 不使用运动量阈值、阶段时长或固定秒数决定裁剪。阶段、动作或接触标记变化立即停止边界裁剪。
- 保持 10 Hz、7 维状态/动作、global 与左腕 wrist 双相机；源数据元信息不一致时停止并报告。
- 不覆盖或修改原始数据、旧训练集和模型。仅生成训练集，本计划不启动训练或上传数据。
- 不自动丢弃异常轨迹或忽略失败；列出问题并停止发布最终训练集，完整集检查通过后才交付。

## 输入与输出

根目录：`/home/lp531/smolvla_data/`。

| 输入目录 | 本次检查轨迹数 | 原始帧数 |
|---|---:|---:|
| z1_tabletop_raw_left_wrist_v1（只选 beverage_bottle） | 184 | 47149 |
| z1_tabletop_raw_bottle_recovery_left_v1 | 22 | 6836 |
| z1_tabletop_raw_bottle_recovery_right_v1 | 20 | 6092 |
| z1_tabletop_raw_bottle_recovery_high_v1 | 20 | 5696 |
| 合计 | 246 | 65773 |

执行开始时重新统计并冻结源文件清单和校验值；以上为计划时快照，不能把它们写成永久常量。

- 新准备目录：`/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1`
- 新 LeRobot 目录：`/home/lp531/smolvla_data/z1_tabletop_lerobot_bottle_recovery_boundary_only_v1`
- repo-id：`lp531/z1_tabletop_bottle_recovery_boundary_only_v1`（仅本地标识）。
- 任何输出目录已存在时拒绝覆盖。
- 准备目录内 `selection_manifest.json` 保存 source_root、source_episode、新编号、源帧数、start、end、首尾删除数、保留帧数、来源类型、源校验值、审核结果。
- `verification_report.json` 保存分组条数、裁剪前后帧数、异常及中间帧连续性验证结果。

## Review Focus

1. 中间静止或重复段：输出必须逐帧保留。
2. 图像相同但动作/标记不同：不能作为边界重复帧删除。
3. 完全重复、单帧或无法识别边界：不生成空轨迹，异常需显式报告。
4. 四个来源 episode 编号重名：新编号唯一，清单能够还原来源。
5. 缺图、无效状态、帧数不符或源数据发生变化：停止并保留诊断，不产生宣称完整的训练集。

## Task 1：准备连续轨迹与裁剪清单

**Files:**
- Create: `/home/lp531/prepare_bottle_recovery_boundary_only.py`
- Test: `/home/lp531/test_bottle_recovery_boundary_only.py`
- Reuse: `/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py` 的 read_frames、load_rgb。

**Interfaces:**
- `boundary_range(frames: list[dict], episode_dir: Path) -> tuple[int, int]` 返回唯一连续区间 [start, end)。
- 脚本消费四个原始目录，产生符合现有导出器输入格式的新准备目录和 selection_manifest.json。

- [ ] 写一个小测试文件，覆盖前后重复压缩、中间重复保留、状态/动作/图像/phase/接触标记变化、单帧、全重复及缺字段保守处理；先运行确认未实现时失败。
- [ ] 实现保守边界判定；测试夹具 A,A,B,C,C,D,D 的结果必须是 [1,6)，即 A,B,C,C,D，保留中间两个 C。
- [ ] 检查所有来源的任务成功与物理抓取标记、每帧两路图像可解码、7 维有限状态/动作、原始连续帧号、单调时间及帧数匹配。阶段标记仅用于审核和禁止跨阶段边界裁剪。
- [ ] 按来源目录固定顺序及原始 episode 编号排序，只选瓶子；分配新的连续 episode 编号，避免重名。用符号链接复用原图，写新的 frames.jsonl 和 summary；仅重编帧号和相对时间，保留原始时间等来源信息于清单/审计记录。
- [ ] 更新新 summary 的 frame_count、duration_s 和新 episode 编号；冻结清单，记录每条裁剪依据。可疑轨迹形成报告，阻止最终转换。
- [ ] 运行 `python3 /home/lp531/test_bottle_recovery_boundary_only.py`，要求全部通过。

## Task 2：转换并逐条核验

**Files:**
- Reuse: `/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py`（无中间裁剪逻辑）。
- Modify: `/home/lp531/test_bottle_recovery_boundary_only.py`（增加清单及来源映射核验）。
- Create: `/home/lp531/verify_bottle_recovery_boundary_only.py`

**Interfaces:**
- 验证脚本读取四个来源、准备目录清单及最终 LeRobot 数据集，输出 verification_report.json；不一致以非零状态退出。

- [ ] 增加测试：多个来源相同 episode 名仍可追溯；缺图、帧数不符、源文件变化和非法中间删除均被拒绝。
- [ ] 执行准备脚本；审核统计及每类首尾裁剪分布，并抽看三类纠偏和正常示范的首尾边界。没有完全重复帧时保持原长度。
- [ ] 在 unifolm-vla 环境调用现有导出器，传入新准备目录、新输出目录、新 repo-id、`--target-object beverage_bottle --camera global --camera wrist`；不使用旧追加脚本。
- [ ] 验证每条保留源索引严格等于 range(start,end)，内容与原始对应帧一致；总保留帧数 = 原始总帧数 - 首部删除总数 - 尾部删除总数。
- [ ] 用 LeRobotDataset 重新打开数据，核对轨迹数、逐条帧数、状态/动作值、10 Hz、相机/状态特征及视频可解码、帧数和顺序一致。视频编码有损，不以像素完全一致验证编码后图像；编码前图像对应关系须精确核验。
- [ ] 确认源校验值未变化，测试通过，产出统计报告与训练集路径。最终数据集应包含冻结清单中的全部正常及纠偏示范；本次快照预期 246 条，最终帧数由合法边界裁剪结果决定。

## 交付

交付新的本地 LeRobot 训练集、来源映射清单及验证报告。报告明确列出每类条数、首尾删除量和“所有中间帧完整保留”的核验依据。训练参数与新旧模型评测另行安排。

## 执行结果（2026-09-30）

数据集已生成并验证：246 条、65,773 帧，两路相机各完整解码 65,773 帧；首尾删除均为 0，中间帧全部保留。10 项专项测试通过，独立审查发现的验证缺口已修复。按用户后续指令，转换验证通过后沿用原双相机 SmolVLA 配置，以 batch_size=8、steps=30000 完成一次训练，15:31:31 核验结束，最终日志 loss=0.023。

最终模型：`/home/lp531/smolvla_training/smolvla_bottle_recovery_boundary_only_batch8_20260930/checkpoints/030000/pretrained_model`。完整记录：`/home/lp531/z1_queue_logs/bottle_recovery_boundary_only_20260930`。尚未进行策略抓取评测。
