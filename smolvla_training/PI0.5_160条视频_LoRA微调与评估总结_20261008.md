# PI0.5：160 条示范数据集、LoRA 微调与自主评估总结

整理日期：2026-10-08；时间按本机北京时间（Asia/Shanghai）记录。

本文对象是 2026-10-07 完成的三相机、二值夹爪、Attention + FFN LoRA 模型，最终检查点为 `080000`。本文不混入旧 246 条数据模型、连续夹爪模型或后来的四条示范实验。内容来自本机保存的数据元信息、实际检查点配置、训练日志、逐次评估结果及当前控制代码。

## 1. 实验概况与当前结果

任务是在 Isaac Sim 中让 Unitree Z1 从桌面物理抓取饮料瓶，抬离桌面，搬运至盒子上方并释放，使瓶子稳定落入盒内。模型输入是三路 RGB、当前七维关节状态和任务文本；输出是六个关节的绝对目标角度及夹爪开闭编码。

| 项目 | 内容 |
|---|---|
| 训练数据 | 160 条成功示范，37,557 个同步时间步，10 Hz |
| “160 条视频”的口径 | 160 条 episode，每条有三路相机；不是仅 160 个单视角视频文件 |
| 图像总量 | 三路合计 112,671 帧；LeRobot 将多 episode 合并存储为视频文件 |
| 训练 / 验证 | 160 / 0 条，本轮没有独立验证集 |
| 底座 | 本地 `pi05_base`；语言骨干 gemma_2b，动作专家 gemma_300m |
| 微调 | 语言模型和动作专家 Attention + FFN LoRA，rank/alpha=32/32 |
| 参数更新 | 80,000 microsteps / 10,000 次优化器更新，有效 batch=8 |
| 完成 | 2026-10-07 11:01；训练进度计时约 10 小时 14 分钟 |
| 最后打印训练 loss | 0.013；不是验证 loss，也不是任务成功率 |
| 本机可核实自主评估 | 7 次有完整结果，成功 0 次；6 次手指碰桌、1 次 60 仿真秒超时未抓取 |
| 用户当前反馈 | 成功率依旧为 0；未提供后续所有测试的总次数，本文不将其并入上述 7 次分母 |

最终模型目录：`/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints/080000/pretrained_model`。

成功示范的验收记录代表示教采集成功；自主评估结果代表模型控制成功，两者是不同指标。仅在训练集输入上预测正确，或参考轨迹回放成功，也不能计入自主成功率。

## 2. 训练集介绍与参数

### 2.1 来源、任务和存储

| 项目 | 实际内容 |
|---|---|
| 机器人 | Unitree Z1，六自由度机械臂及夹爪 |
| 数据来源 | 仿真示教 / 原生抓取流程记录；包含接近、闭爪、试抬升、搬运、释放及成功保持 |
| 原始数据 | `/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1` |
| 连续夹爪转换副本 | `/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_20261006` |
| 本轮实际训练集 | `/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007` |
| repo_id | `lp531/z1_bottle_three_camera_pi05_binary_20261007` |
| 数据格式 | LeRobot v3.0；数值数据为 Parquet，RGB 为 AV1/yuv420p MP4 |
| 任务数量 | 1，目标物为 beverage_bottle；其他桌面物体为场景干扰物 |
| 文本 | `Physically grasp the bottle, lift it clear, and place it in the box.` |
| 采样 | 10 Hz，时间步间隔 0.1 s |
| episode 长度 | 最短 189 帧，长度中位数 223.5 帧，最长 509 帧，即约 18.9–50.9 s |
| 总时长 | 37,557 / 10 = 3,755.7 s，约 62.6 分钟；三相机是同时采集，时长不乘三 |
| 补录过程 | 原保留 125 条，再补 35 条成功示范至 160 条 |
| 数据保留 | 二值转换未删除首尾或中间帧；前六维动作、状态、时间戳及视频字节保持一致 |

源三相机版本曾记录 140/20 的划分；**本轮二值版本实际使用全部 160 条训练，eval_split=0，未继承 140/20 划分**。早期转换报告里的“未启动训练”是当时状态，后续真实训练日志已确认完成。

### 2.2 空间覆盖与示范姿态

场景全工作区随机化配置为 x∈[0.36,0.69] m、y∈[−0.27,0.16] m，划分为 5×5 网格。本数据只保留行索引 0–3，行索引 4 未采集；每个保留格 8 条，20 格共 160 条。初始瓶子位置从最终清单统计为 x∈[0.360397,0.689943] m、y∈[−0.268464,0.063707] m。这里的位置是场景物体参考点，不是瓶子中心高度。

| 网格行索引 | 列 0 | 列 1 | 列 2 | 列 3 | 列 4 | 小计 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 8 | 8 | 8 | 8 | 40 |
| 1 | 8 | 8 | 8 | 8 | 8 | 40 |
| 2 | 8 | 8 | 8 | 8 | 8 | 40 |
| 3 | 8 | 8 | 8 | 8 | 8 | 40 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 |

160 条均属于工具滚转约 −90° 的侧向夹持示范。已有审计的实际滚转范围为 −91.68°～−88.41°，中位数 −89.81°。按 2 mm / 1° 静止判据，闭爪前静止时段约 1.1–1.5 s，中位数 1.3 s。成功记录均有物理抓取、抬离桌面、释放及稳定放置证据；这不保证每条示范在当前修改后的执行条件下都可重放成功。

审计没有重复 seed 或重复轨迹，但记录了 **387 项 action_joint_limit 越界项**。它是审计违反项计数，不能直接解释为 387 条轨迹。本轮转换保留原前六维标签，没有据此修正标签。需要关注示教目标与模型执行限位之间的差异。

### 2.3 观测、动作和相机

| 字段 | 规格与含义 |
|---|---|
| observation.state | float32，[7]：joint1…joint6、jointGripper，均为实际角度 rad |
| action | float32，[7]：前六维为绝对关节目标 rad；第七维为 gripper_close_binary |
| observation.images.wrist | 左腕 RGB，640×360；topic 为 `/left_wrist_camera/color/image_raw` |
| observation.images.global | 外部全局 RGB，640×480；topic 为 `/global_camera/color/image_raw` |
| observation.images.global_right | 另一侧外部 RGB，640×480；topic 为 `/global_right_camera/color/image_raw` |
| 模型图像配置 | 224×224，经 PI05 图像预处理；不是把源视频改成 224×224 |
| 额外数据列 | timestamp、frame_index、episode_index、index、task_index |
| 视频编码 | AV1，yuv420p，10 fps，无音频；meta 记录 crf=30、preset=12、g=2、PyAV |
| 模型未使用的信息 | 深度图、瓶子真值坐标、接触状态和任务阶段不作为本轮七维观测输入 |

采集同步方法为固定时间网格、最近 RGB、插值关节状态和因果动作指令配对；同步容差 45 ms，缓存延迟 0.5 s，允许动作年龄 0.40 s。动作是已发送的目标指令，不是用事后实际关节角冒充动作标签。缓存延迟用于等齐数据，不表示给模型增加了 0.5 s 历史输入。

### 2.4 二值夹爪标签

夹爪**动作标签**为 0=开、1=闭；夹爪**观测状态**仍是实际角度。例如被瓶子挡住时，闭爪动作保持 1，实际关节可能停在中间角度。

标签根据原始因果动作和阶段重建：READY 标 0；GRASPING 原始闭爪指令首次大于 −0.895 rad 时起标 1；闭合、接触保持、试抬升和搬运保持 1；最终连续开爪斜坡的第一次下降指令起标 0，之后释放和成功保持为 0。原始数据没有显式开闭意图字段，这些是重建意图标签。

转换统计：开爪 17,795 帧（47.38%），闭爪 19,762 帧（52.62%）。每条正好一次开转闭、一次闭转开。相较简单按 −0.75 rad 阈值分类，1,082 帧标签不同。动作第七维 q01=0、q99=1。

部署必须在反归一化后将第七维按 0.5 阈值映射：≥0.5→闭爪目标 0 rad，<0.5→开爪目标 −0.90 rad，再应用执行限位和限步。模型原始输出可以略超出 0–1，阈值判断仍适用。状态观测不能同步改成 0/1。

### 2.5 归一化与数据处理

状态和动作采用 QUANTILES，统计从全部 160 条数据重算。公式为 `x_norm = 2 × (x − q01) / (q99 − q01) − 1`，推理后动作反归一化。图像采用 VISUAL=IDENTITY，部署 RGB 张量缩放到 0–1。图像增强关闭；没有阶段加权或独立验证拆分。完整 min/max、mean/std 与分位数见配套证据快照，文档末尾另列状态和动作范围。

## 3. PI0.5 LoRA 微调介绍与参数

### 3.1 模型与训练方法

PI05Policy 将图像、任务文本和当前状态用于条件动作生成。本地实现使用流匹配：对目标动作块加噪，训练模型预测对应向量场；推理从噪声迭代生成动作块。本轮训练的是动作预测损失，没有直接优化“成功/失败”奖励，也没有在训练循环中运行仿真成功率评估。

LoRA 为线性层增加低秩可训练分支，主体底座权重保持冻结。本轮扩展到语言模型与动作专家的全部 Attention 投影 q/k/v/o，以及 FFN 的 gate/up/down 投影；另外四个动作 / 时间投影层全量训练并随适配器保存。部署需用底座加载 LoRA，不能把 adapter 权重当作独立完整底座。

| 模型参数 | 实际值 |
|---|---|
| 实现 | 工作区本地 LeRobot PI05Policy + PEFT |
| 底座目录 | `/home/lp531/pi05_lora_workspace/weights/pi05_base` |
| 语言 / 动作专家 | gemma_2b / gemma_300m |
| LoRA rank / alpha / dropout / bias | 32 / 32 / 0.0 / none |
| Attention 覆盖 | 语言 72 层投影模块 + 专家 72 层投影模块 |
| FFN 覆盖 | 语言 54 层投影模块 + 专家 54 层投影模块 |
| LoRA 总模块数 | 252；各骨干 18 个 Transformer 层 |
| 完整训练层 | action_in_proj、action_out_proj、time_mlp_in、time_mlp_out |
| 可训练参数 | 55,249,952，约 5,525 万；来源为真实模型结构的 PEFT meta 检查 |
| 视觉编码器 | freeze_vision_encoder=true |
| 仅专家训练 | train_expert_only=false；本轮不仅微调动作专家 |
| 观测历史 | n_obs_steps=1；视觉记忆、本体记忆均关闭 |
| chunk_size | 15，预测 15×7 动作，按训练 10 Hz 对应约 1.5 s |
| n_action_steps | 配置为 5；独立仿真 rollout 的实际窗口由 execute_steps 决定 |
| 内部最大状态 / 动作维度 | 32 / 32；本任务有效维度均为 7 |
| 相对动作 | use_relative_actions=false |
| 推理迭代 | num_inference_steps=10 |
| tokenizer | 本地 weights/tokenizer；最大文本长度 200 |
| 精度 | bfloat16，accelerator mixed_precision=bf16，policy.use_amp=false |
| 其他 | gradient_checkpointing=true；compile_model=false；EMA=false |

LoRA 目标正则：

```text
.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)
```

流匹配时间采样参数：Beta alpha=1.5、beta=1.0、scale=0.999、offset=0.001；时间编码 min_period=0.004、max_period=4.0。

### 3.2 优化器与训练规模

| 参数 | 实际值 |
|---|---|
| 初始化 | 官方底座本地权重；新建 LoRA，resume=false，不从旧连续夹爪 adapter 续训 |
| microbatch | 1 |
| 梯度累积 | 8 |
| 数据并行 | 1 |
| 有效 batch | 1×8×1=8 |
| steps | 80,000 microsteps |
| 优化器更新 | 80,000÷8=10,000 次 |
| 样本窗口总量 | 80,000，约为 37,557 个锚点的 2.13 倍；不是每条示范精确重复同一次数 |
| 优化器 | AdamW |
| 峰值 / 末端学习率 | 5e-5 / 5e-6 |
| 调度 | cosine_decay_with_warmup |
| warmup | 4,000 microsteps，500 次参数更新 |
| decay 配置长度 | 80,000 microsteps |
| betas / eps | (0.9,0.95) / 1e-8 |
| weight_decay / 梯度裁剪 | 0.01 / 范数 1.0 |
| seed | 20261007 |
| DataLoader | num_workers=4，prefetch_factor=2，persistent_workers=true，spawn，PyAV |
| 保存 / 日志频率 | 8,000 / 80 microsteps，分别对应 1,000 / 10 次更新 |
| 验证 | eval_split=0、eval_steps=0、env_eval_freq=0 |
| 上传与追踪 | WandB 关闭，Hub 上传关闭；本地 safetensors 保存 |

本地训练日志的 step 是 microstep，不能把 80,000 直接写成 80,000 次优化器更新。调度配置口径与本地训练器一致。

### 3.3 训练完成与模型选择

启动验证记录时间为 2026-10-07 00:47；最终日志在 11:01:00 写出 `Checkpoint policy after step 80000` 与 `End of training`。最后打印 loss=0.013、lr=5e-6、mem_gb=9.97。这里的显存是日志记录值，不是 GPU 总显存。

保存检查点为 008000、016000、024000、032000、040000、048000、056000、064000、072000、080000，last 指向 080000。最终 adapter_model.safetensors 大小 221,091,648 bytes，约 210.85 MiB。

本轮选用的是训练完成的最后检查点，没有独立验证 loss 支持“验证最优”选择。保存配置、adapter_config 与部署入口均确认所用模型是 080000。已有离线接口检查能输出 15×7 动作并正确做二值夹爪映射，但单帧接口通过不能说明自主抓取成功。

## 4. 评估场景介绍

### 4.1 环境与传感器

环境为 Isaac Sim + ROS 2 Humble 的 Z1 桌面场景，包含桌面、目标瓶子、接收盒和 mouse/cup/apple/peach 等干扰物。三相机需以 `external_camera_layout:=dual` 启动，模型使用左腕和两个外部视角。深度和额外 VLA 腕部相机即使在资产中存在，也不是这个模型的三路输入。

| 场景配置 | 内容及证据口径 |
|---|---|
| 坐标 | 场景资产以 Z1 link00 为参考，+X 向前、+Z 向上；桌面 top_z=0 m |
| 目标瓶子 | cylinder_side_pinch；配置半径 0.0242 m，高 0.157 m |
| 资产瓶子质量 | generated 场景元信息为 0.18 kg；逐次结果未保存实际运行时质量快照 |
| 全局相机位置 | 当前 dual 配置 global=(0.98,−0.56,0.74) m，global_right=(0.98,0.56,0.74) m |
| 外部相机朝向 | 均朝向 (0.5,−0.055,0.1) m，640×480；fx=fy≈554.256，cx=320、cy=240 |
| 腕部相机 | 模型订阅左腕 640×360 RGB；旧资产里的另一 VLA 腕部安装参数不应当作该输入的安装参数 |
| 随机化 | 已完成评估记录为 full；随机位置、朝向、其他物体布局，再做静置验收 |
| 工作区配置 | x=[0.36,0.69] m，y=[−0.27,0.16] m，full yaw_delta=180° |
| 布局约束 | 最小物体间距 0.02 m；目标接近 / 抬升扫掠区域需通过检查 |
| 初始静置验收配置 | 至少 0.9 s；水平漂移≤4 mm、垂直漂移≤6 mm、速度≤0.01 m/s、角速度≤0.2 rad/s、姿态漂移≤5°、桌面穿透≤3 mm，要求触桌 |
| 距训练布局约束 | 当前 SH 限定目标距最近训练示范位置≤0.03 m；不是严格复用整幅训练场景 |
| 瓶子冻结 | 这 7 次记录 freeze_target_until_close=false，自由物理状态、碰撞开启 |
| 抬升前等待 | prelift_contact_stable_s=0，此额外控制门控关闭；物理成功判据仍有效 |

上述相机外参与未记录在 result 中的物理参数是**当前代码 / 配置值**。旧 generated 场景中的单外部相机坐标不是 dual 布局值。逐次评估没有完整保存质量、摩擦、所有相机外参和启动参数快照，因此不能保证仅靠结果 JSON 重建历史全部物理设置；配置文件副本随证据快照保存。

### 4.2 成功与失败判据

完整成功需要先有已验证的物理抓取，完成抬升搬运和释放，物体参考点处于盒内 XYZ 成功区域，速度≤0.03 m/s 且持续稳定至少 0.5 s，同时没有锁存失败原因。仅靠夹爪接近瓶子、瓶子高度升高或某帧在盒子上方，不能判定成功。

当前评估器默认盒中心 (0.32,0.14) m，内半宽 0.055 m，z∈[0,0.12] m；对应参考点范围 x∈[0.265,0.375]、y∈[0.085,0.195] m。这是评估器参数默认值，若启动覆盖参数则以覆盖值为准。该判据检验参考点，不是证明整个瓶子网格都包含在盒内。

物理抓取检查结合实际双指接触、抬离桌面、相对位姿与稳定性；瓶子配置的证明抬升至少 0.03 m、桌面净空至少 0.02 m。抓取验收区在夹爪局部坐标中为 x=0.110–0.154 m、|y|≤0.040 m、z=0.006–0.030 m。模型不能读取这些真值来直接生成动作，评估器用它们验收。

运行期间遇到手指触桌、跟踪异常、输入过期、操作者停止或超时会终止。碰桌停止是在接触发生后响应，不是提前路径避碰。

## 5. 评估控制流程与参数

### 5.1 模型如何控制机械臂

流程为：重置机械臂 → 随机化并验收场景 → 同步三路 RGB 和关节状态 → PI05 预处理、预测及反归一化 → 取执行窗口 → 映射夹爪开闭 → 关节限位与限步 → 60 Hz 插值发布 `/isaac_joint_commands` → 检查接触与执行状态 → 下一轮观测和推理。

模型输出绝对关节目标，rollout 不调用 MoveIt 为每段预测生成避碰路径。示教流程中的规划避碰能力不会自动成为模型执行能力。60 Hz 是动作发布频率目标，不是 60 Hz 视觉模型推理；动作时间尺度仍为 10 Hz，每个 0.1 s 动作段插值成六个子步。

每次模型预测 15 点，已保存的 7 次评估实际取前 **10 点**，即约 1 s 的名义动作跨度，执行完才重新视觉推理，剩余 5 点丢弃。当前模型选择 SH 则默认 **5 点**，名义跨度约 0.5 s；可由命令行覆盖。推理耗时及仿真持续运行会影响实际闭环间隔，不能将名义跨度写成精确实测重规划周期。

### 5.2 控制参数

| 参数 | 7 次历史结果 / 当前代码口径 |
|---|---|
| policy_type | pi05；checkpoint=080000 |
| action_rate | 10 Hz |
| execution_rate | 60 Hz，六插值子步 / 动作段 |
| execute_steps | 历史 7 次=10；当前 SH 默认=5 |
| 预测块 | 15 点；配置 n_action_steps=5 不强制独立 rollout 也取 5 点 |
| binary_gripper | true；gripper_action_encoding=binary |
| 模型阈值 | 反归一化第七维≥0.5→0 rad，否则→−0.90 rad |
| 物理夹爪状态阈值 | −0.75 rad 为已命令闭爪等状态逻辑的角度阈值，不是模型二值输出阈值 |
| 单次限步 | 当前代码默认 max_joint_step=0.10 rad，相对实际关节状态应用 |
| 跟踪故障 | 默认误差>0.08 rad 持续 0.5 s 终止 |
| 遥测过期 | 默认 stale_timeout=0.5 s |
| 输入同步 | 结果记录 tolerance=0.025 s，max_age=0.2 s；等待超时当前默认 0.5 墙钟秒 |
| 最大 episode 时间 | 60 仿真秒 |
| 抬升接触门控 | stable_s=0，关闭；timeout_s 默认 2.0，此轮不触发该门控 |
| 目标固定 | 关闭，非磁吸 / 非冻结目标 |
| 相机检查 | 当前 SH 启用 reject_blank_cameras |

执行限位如下；rad 单位数值在代码中由角度转换：

| 关节 | 最小角度 | 最大角度 |
|---|---:|---:|
| joint1 | −150° | 150° |
| joint2 | 0° | 170° |
| joint3 | −165° | 0° |
| joint4 | −87° | 87° |
| joint5 | −77° | 77° |
| joint6 | −160° | 160° |
| jointGripper | −90° | 0° |

限位之后再将目标相对当前实际位置的变化限制到±0.1 rad，插值子步也经过相应检查。闭爪映射到 0 rad 不代表电机一步到 0 rad，更不意味着夹爪被瓶子阻挡时会实际闭到 0 rad。模型预测超限时执行动作会改变，原始预测、限位后指令与实际反馈必须区分。

### 5.3 当前入口与历史条件复现

已有仿真场景须启用 dual 外部相机布局；以下命令只是记录入口，本文整理过程中未执行机器人控制。

```bash
# 当前 SH 默认：本模型、自由瓶子、10 Hz 动作、60 Hz 插值、执行 5 点
bash /home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh --model pi05

# 覆盖为历史结果使用的 10 点窗口；其他历史物理条件仍需核对
bash /home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh --model pi05 --execute-steps 10
```

## 6. 已保存自主评估结果与现象

下表从本模型目录全部 result.json 逐次统计；各 report.json 的 checkpoint 均为 080000。无完整 result 的目录不计入分母。

| 2026-10-07 开始时间 | seed | 完整成功 | 结果 | 仿真时长 |
|---|---:|---|---|---:|
| 12:11:58 | 267632139 | 否 | finger_table_contact | 16.483 s |
| 12:13:48 | 446181338 | 否 | finger_table_contact | 26.400 s |
| 12:15:57 | 489361075 | 否 | finger_table_contact | 6.633 s |
| 12:21:22 | 874416086 | 否 | grasp_failed，超时未抓取 | 60.000 s |
| 12:24:28 | 774196979 | 否 | finger_table_contact | 6.783 s |
| 12:26:01 | 1126814434 | 否 | finger_table_contact | 16.850 s |
| 12:34:14 | 277459214 | 否 | finger_table_contact | 19.533 s |

合计 **0/7=0%**。原 10 月 7 日碰桌分析只统计了前六次（0/6），本文扫描到后来保存的 12:34:14 结果，更新为七次；前六次几何分析不能未经检查直接套用到第七次。

已有前六次分析显示：两次在约 6.6/6.8 仿真秒、夹爪仍张开时碰桌；另一些轮次 joint5 预测超过执行上限，裁剪改变姿态与毫米级桌面余量。12:13:48 曾发出闭爪，但当时已偏离抓取位置。前六次同步日志 skew=0，实际关节大体跟随已发布指令，不能据此把失败都归因于传感器错配或无法执行指令。

这些证据支持当前主要失败表现为接近路径和末端姿态不准确，部分轮次叠加执行限位裁剪；尚未证明学习率、LoRA rank、训练步数、数据量、腕部遮挡或 10 点窗口中的任一项是唯一根因。低训练 loss、160 条成功示范及空间上接近训练位置，都不足以保证安全闭环抓取。

## 7. 对照时应保留的实验差异

| 差异 | 解释 |
|---|---|
| 示教成功 vs 自主成功 | 原生示教有规划和阶段控制；模型以预测关节目标直接执行 |
| 完整数据保留 vs 标签无问题 | 保留数据不等于所有标签满足当前执行限位；审计有越界项 |
| 训练 15 点 vs 执行 10/5 点 | 预测长度、独立执行窗口和配置 n_action_steps 是三个不同字段 |
| 10 Hz vs 60 Hz | 10 Hz 是动作点时间尺度；60 Hz 是插值发布频率，不是运动加速六倍 |
| 二值意图 vs 实际夹爪角度 | 模型输出 0/1，物理反馈仍为角度；接触会阻止关节到达理想闭爪角度 |
| 全 160 条训练 vs 留出评估 | 本轮无验证集，不能将训练 loss 当作泛化验证 |
| 自由瓶子 vs 固定瓶子诊断 | 本轮自主评估自由碰撞；其他 assisted / replay 实验不计入成功率 |
| 代码默认值 vs 历史实测 | result 有记录的值优先；未保存完整启动快照的物理设置明确列为配置来源 |

## 8. 证据来源与附录

以下本地文件是本文依据；路径指向原始产物。配套 JSON 快照保存本轮完整训练、adapter、数据及场景配置，CSV 清单列出全部 160 条示范、原始 episode、网格、seed、长度和初始目标位置，便于复核。

- [数据集元信息](/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007/meta/info.json)
- [数据归一化统计](/home/lp531/smolvla_data/z1_bottle_three_camera_pi05_binary_20261007/meta/stats.json)
- [最终示范清单](/home/lp531/pi05_lora_workspace/three_camera_20261006/split_manifest.json)
- [二值标签转换检查](/home/lp531/pi05_lora_workspace/binary_gripper_20261007/conversion_audit.json)
- [训练集严格审计](/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006/summary.json)
- [最终训练配置](/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints/080000/pretrained_model/train_config.json)
- [最终 LoRA 配置](/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints/080000/pretrained_model/adapter_config.json)
- [训练日志](/home/lp531/pi05_lora_workspace/binary_gripper_20261007/training.log)
- [PEFT 结构检查](/home/lp531/pi05_lora_workspace/binary_gripper_20261007/training_setup_verification.json)
- [本模型评估目录](/home/lp531/smolvla_eval/pi05_bottle_binary_chunk15_attn_ffn)
- [前六次碰桌分析](/home/lp531/smolvla_eval/pi05_binary_table_review_20261007/评估分析.md)
- [模型选择入口](/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh)
- [rollout 控制代码](/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/smolvla_sim_rollout.py)
- [限位及夹爪转换代码](/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/smolvla_rollout_core.py)
- [物理与随机化配置](/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json)
- [dual 外部相机配置](/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/external_cameras.json)
- [成功判据代码](/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/grasp_evaluator.py)

### 8.1 状态和动作统计

下表直接取实际训练集 stats.json，角度保留六位小数；第七维动作是二值，无 rad 单位。

#### 观测状态（rad）

| 维度 | min | max | mean | std | q01 | q99 |
|---|---:|---:|---:|---:|---:|---:|
| joint1 | -1.363300 | 1.120500 | -0.109880 | 0.515886 | -1.071008 | 1.066800 |
| joint2 | 0.003700 | 2.694800 | 1.648284 | 0.607146 | 0.003800 | 2.557044 |
| joint3 | -1.827300 | 0.000400 | -0.673669 | 0.371788 | -1.588376 | 0.000300 |
| joint4 | -1.513600 | 1.500200 | -0.696882 | 0.533267 | -1.449500 | 1.270944 |
| joint5 | -1.338200 | 1.341000 | 0.178998 | 0.770176 | -1.298800 | 1.316744 |
| joint6 | -2.652300 | 0.769800 | -1.288980 | 0.611379 | -2.242752 | 0.395800 |
| jointGripper | -0.933200 | -0.639000 | -0.791428 | 0.093753 | -0.926500 | -0.668700 |

#### 动作（前六维 rad，第七维 0/1）

| 维度 | min | max | mean | std | q01 | q99 |
|---|---:|---:|---:|---:|---:|---:|
| joint1 | -1.362725 | 1.120550 | -0.109843 | 0.516262 | -1.070585 | 1.066777 |
| joint2 | 0.003200 | 2.692423 | 1.647613 | 0.605156 | 0.003200 | 2.555887 |
| joint3 | -1.832855 | 0.000300 | -0.679172 | 0.370491 | -1.592608 | 0.000300 |
| joint4 | -1.517296 | 1.500021 | -0.702460 | 0.533581 | -1.454500 | 1.269844 |
| joint5 | -1.343742 | 1.342591 | 0.179298 | 0.772239 | -1.303963 | 1.318792 |
| joint6 | -2.650226 | 0.771608 | -1.289237 | 0.610057 | -2.240188 | 0.399449 |
| gripper_close_binary | 0.000000 | 1.000000 | 0.526187 | 0.499314 | 0.000000 | 1.000000 |

配套文件：

- [PI0.5_160条视频_LoRA微调与评估总结_20261008_证据快照.json](/home/lp531/smolvla_training/PI0.5_160条视频_LoRA微调与评估总结_20261008_证据快照.json)
- [PI0.5_160条视频_LoRA微调与评估总结_20261008_160条示范清单.csv](/home/lp531/smolvla_training/PI0.5_160条视频_LoRA微调与评估总结_20261008_160条示范清单.csv)
