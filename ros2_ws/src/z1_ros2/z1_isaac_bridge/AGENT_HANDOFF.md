# Z1 SmolVLA 训练后端 Agent 交接文档

最后核对：2026-09-10（Asia/Shanghai）

本文档是后续 Agent 接手本项目时的首要入口。若本文档与旧版 `README.md`
或历史工作日志冲突，以当前源码、`config/grasp_physics.json` 和本文档注明的
“当前状态”为准。

## 1. 项目目标

本项目在 Isaac Sim 中控制 Unitree Z1 机械臂，通过 ROS 2 Humble、MoveIt 2 和
纯物理抓取判定，自动生成 SmolVLA/LeRobot 可用的桌面抓取放置示教数据。

当前任务是从桌面抓取指定目标，并将其自然释放到蓝色盒子中。支持目标：

- `cup`
- `apple`
- `peach`
- `mouse`
- `beverage_bottle`

训练数据由确定性专家流程生成，不是由 SmolVLA 策略在线推理生成。因此，采集
示教数据时不需要挂 SmolVLA 推理后端。SmolVLA/LeRobot 环境只在后续转换、训练
或策略部署阶段需要。

## 2. 关键路径

| 内容 | 路径 |
| --- | --- |
| ROS 工作区 | `/home/lp531/ros2_ws` |
| 本 ROS 包 | `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge` |
| 统一物理与随机化配置 | `config/grasp_physics.json` |
| 一键自动示教编排 | `z1_isaac_bridge/grasp_gui_bridge.py` |
| 连续采集脚本 | `z1_isaac_bridge/continuous_training.py` |
| 抓取与搬运专家 | `z1_isaac_bridge/grasp_assistant.py` |
| 成功判定 | `z1_isaac_bridge/grasp_evaluator.py` |
| Episode 录制 | `z1_isaac_bridge/episode_recorder.py` |
| 时间同步器 | `z1_isaac_bridge/recording_sync.py` |
| Episode 复位 | `z1_isaac_bridge/episode_reset.py` |
| 静态碰撞场景 | `z1_isaac_bridge/planning_scene_sync.py` |
| 动态物体碰撞同步 | `z1_isaac_bridge/object_collision_sync.py` |
| ROS/Isaac 启动文件 | `launch/z1_isaac.launch.py` |
| Isaac 场景运行脚本 | `/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py` |
| 当前原始数据集 | `/home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist` |
| LeRobot 转换器 | `/home/lp531/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py` |
| ROS 日志 | `/home/lp531/.ros/log` |

## 3. 系统架构与数据流

```text
continuous_training CLI / Web GUI
                |
                v
grasp_gui_bridge:8767  ---- GET /api/status
                |
                +--> episode_reset --> Isaac 场景复位/随机化
                +--> grasp_assistant --> MoveIt 规划/轨迹执行/闭爪/搬运/释放
                +--> grasp_evaluator --> 物理接触、抬升与稳定放置判定
                +--> episode_recorder --> 三路 RGB + 关节状态 + 动作 + 标签
                                          |
                                          v
                         z1_tabletop_raw_dual_wrist/episode_NNNNNN
                                          |
                                          v
                               LeRobot 数据集转换与训练
```

Isaac 发布 `/clock`、`/joint_states`、物体真值、接触状态和相机。MoveIt 负责
碰撞安全规划。轨迹桥将 FollowJointTrajectory 转发给 Isaac。抓取过程中不存在
“物体附着”“物体跟随夹爪”或传送约束；抓取与搬运必须完全依靠重力、摩擦和真实
双指碰撞接触。

## 4. 标准启动方式

### 4.1 构建

本机用户目录中的 setuptools 84 与 ROS Humble 的 colcon Python 扩展不兼容，
会出现 `--editable` 或 `--uninstall` 不识别。构建时必须屏蔽用户 site-packages：

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
PYTHONNOUSERSITE=1 colcon build --symlink-install \
  --packages-select z1_isaac_bridge
source install/setup.bash
```

### 4.2 启动训练栈

正式批量采集推荐关闭 RViz，降低渲染和 CPU 压力：

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py rviz:=false
```

调试规划时可保留默认 `rviz:=true`。GUI 地址：

```text
http://127.0.0.1:8767
```

严禁同时启动两套 `z1_isaac.launch.py`。重复栈会产生两个 Isaac、两个 MoveIt、
重复同名 ROS 节点和两个 `/clock` 发布者，造成时间倒退、TF 清空、RViz 崩溃、
控制命令竞争以及两个录制器同时写同一个数据集。

启动前建议检查：

```bash
pgrep -af "ros2 launch z1_isaac_bridge|run_z1_ros2_scene.py|continuous_training"
ros2 topic info /clock --verbose
```

`/clock` 应只有一个发布者。不要同时运行其他启用相机的 Isaac Sim 任务；此前一个
G129 Isaac 进程与 Z1 同时运行，合计消耗大量 CPU/GPU，并导致相机发布抖动。

## 5. 一键自动示教

核心 HTTP 接口：

```http
POST /api/demonstration/auto
Content-Type: application/json

{"target_object":"apple","curriculum_level":"full"}
```

一次自动示教是原子工作流：

1. 设置目标物体。
2. 按课程等级完整复位并随机化场景。
3. 在开始录制前生成预抓取规划。
4. 若规划失败，一直重新随机化布局直到通过（不再设上限，只有“强制停止”能终止）；每个布局评估 8 个候选。
5. 规划成功后开始录制，并执行同一条预抓取轨迹。
6. 闭环对中、插入、慢速闭爪，要求真实双指持续接触。
7. 在重力开启状态下完成物理试抬和带载验证。
8. 搬运到蓝盒，下降并自然释放。
9. 等待稳定放置判定，通过后保存 Episode。
10. 任何异常都会返回失败；若已经开始录制，则失败录制会被丢弃。

单次 GUI 自动示教失败后会故意保留失败姿态供观察，需要手动调用“一键重置
Episode”。连续训练脚本会自动完成这个恢复动作。

常用接口：

- `GET /api/status`：总状态、抓取、录制、碰撞场景、复位和自动示教进度。
- `POST /api/reset_episode`：完整停止、丢弃、物理复位、清状态、刷新碰撞体。
- `POST /api/force_stop`：停止当前自动示教与运动。
- `POST /api/randomize_episode`：按课程等级随机化并完整复位。
- `POST /api/recording/start|save|discard`：手动录制控制。

## 6. 连续“一键训练/采集”脚本

入口：

```bash
ros2 run z1_isaac_bridge continuous_training \
  --target apple \
  --curriculum full \
  --episodes 100
```

参数：

- `--target`：五种受支持物体之一。
- `--curriculum easy|medium|full`：随机化等级，默认 `easy`。
- `--episodes N`：需要成功保存的轮数，不是最大尝试次数。
- `--retry-delay S`：失败复位后等待时间，默认 2 秒。
- `--max-attempts N`：可选安全上限；`0` 表示无限尝试。
- `--host`、`--port`：GUI bridge 地址，默认 `127.0.0.1:8767`。

脚本行为：

- 启动时等待连接、录制器、复位器和碰撞场景全部就绪。
- 每次调用完整自动示教。
- 只有服务返回成功且 Episode 成功保存时，成功计数才加一。
- 任意失败都执行 `force_stop -> discard -> reset_episode`。
- 复位失败时持续重试，绝不会带着上一轮失败状态直接开始下一轮。
- `Ctrl-C` 会强停并丢弃未完成 Episode。

该脚本无需修改即可继承 launch 和录制器的新参数。修改相机、随机化或抓取代码
后，必须停止连续脚本并重启整个 Z1/Isaac 栈。

## 7. 当前课程随机化配置

配置来源：`config/grasp_physics.json -> scene_randomization`。

| 等级 | 位置 | 最小位移 | Yaw |
| --- | --- | --- | --- |
| easy | 相对资产初始位 ±0.060 m | 0.035 m | ±45° |
| medium | 相对资产初始位 ±0.120 m | 0.060 m | ±90° |
| full | 工作区采样 | 0.080 m | ±180° |

全局 Full 工作区仍用于干扰物和尚未设置独立范围的目标：

```text
X = 0.36 .. 0.69 m
Y = -0.27 .. 0.16 m
```

苹果目标已增加独立、保守的 Full 范围：

```text
X = 0.40 .. 0.55 m
Y = -0.23 .. 0.02 m
Yaw = ±180°
```

只有当当前目标是 `apple` 且等级为 `full` 时应用这个范围；其他物体目标暂时仍
使用全局 Full 工作区。干扰物继续使用全局范围，但必须通过物体间距、蓝盒排除、
目标扫掠通道、闭爪区和初始抬升区检查。

苹果范围刚完成修改并通过静态测试，但尚未完成新的长时间成功率统计。不要把范围
缩小视为已解决全部失败；搬运掉落仍是独立问题。

## 8. 数据录制与保存门槛

当前数据根目录：

```text
/home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist
```

每帧包含：

- `observation.images.wrist`：640×360 RGB
- `observation.images.vla_wrist`：640×360 RGB
- `observation.images.global`：640×480 RGB
- `observation.state`：Z1 六轴加夹爪，共 7 维实测关节
- `action`：对应的 7 维命令
- 任务文本、阶段、抓取/放置标签和时间戳

录制设置：

```text
fps = 10
sync_tolerance_s = 0.045
sync_delay_s = 0.5
minimum_frames = 10
require_task_success = true
```

仿真物理与渲染保持 60 Hz，三路 RGB 的 Image 和 CameraInfo helper 使用
`frameSkipCount=0`。2026-09-11 的实测表明，`frameSkipCount=1` 只得到约 14--17 Hz，
并把最长间隔扩大到 0.23 秒以上；把全局渲染步长改为 30 Hz 也会使全局相机降到
约 13.7 Hz，因此两种“直接降频”方案均已撤销。当前更好的配置是发布端
`queueSize=1`、Best-Effort/Volatile QoS：不积压旧的大图像，只保留最新帧。深度流
默认关闭，只有显式启用 `--depth-streams` 时才创建。

同一满栈、录制器在线、640x360/640x360/640x480、30 秒测试中，三路仿真时间频率
约为 36.3/36.2/31.5 Hz，95% 帧间隔约 83.3/66.7/83.3 ms。将全局相机降到
512x384 能进一步平衡频率，但会与现有数据集的图像尺寸元数据不兼容，因此未作为
默认值。录制器现在先按消息时间戳选择 10 Hz 帧，只对入选帧执行 NumPy/OpenCV
转换，避免把每一帧都重复转换。

时间同步器要求每个采样点使用新的相机帧，关节状态可插值，动作必须是目标时刻
之前的最近因果命令。任一路图像在 ±45 ms 内没有新帧，都会锁存
`sync_failure`，该 Episode 不可保存。

保存必须同时满足：

- 没有时间同步失败。
- 至少 10 帧。
- 曾验证真实物理抓取。
- 物体曾离开桌面。
- 曾观测到双指接触。
- 已完成释放序列。
- 物体稳定处于蓝盒成功区域，即 `task_success=true`。

写入流程为：

```text
.recording_NNNNNN  --成功原子重命名-->  episode_NNNNNN
```

`.recording_*` 不属于可训练数据。成功 Episode 还会写入 `episodes.jsonl`。

## 9. 当前数据和运行状态

截至本文档更新时间：

- 没有运行中的 Z1 launch、Isaac Z1 场景或 `continuous_training`。
- 数据集中没有成功保存的 `episode_NNNNNN`。
- 当前没有 `episodes.jsonl`。
- 存在诊断目录 `.recording_000000`，其中有 102 帧。
- 该目录的 `interrupted.json` 表明录制被中断，不能用于训练。
- 该中断布局为苹果 Full，seed `1719494697`；苹果位置约
  `[0.403134, -0.037923, 0.0] m`。

此前用户要求清空数据，旧数据被可恢复地移动到：

```text
/home/lp531/.local/share/Trash/files/
z1_tabletop_raw_dual_wrist_before_clear_2026-09-10
```

不要在未获得用户明确指示时删除回收站备份或当前中断诊断目录。

## 10. 当前进度与最近已完成修改

### 已完成

- 建立完整 Z1 Isaac/ROS/MoveIt 抓取与放置后端。
- 建立五种目标物体和静态桌面/蓝盒、动态物体碰撞同步。
- 建立纯物理抓取、抬升、释放和稳定放置成功判定。
- 建立三路 RGB 同步原始 Episode 录制。
- 建立 GUI 一键自动示教 API。
- 新增 `continuous_training`：失败丢弃并自动复位，直到保存指定成功轮数。
- 修复相机同步余量：容差 45 ms、延迟缓冲 0.5 s；相机发布使用深度 1 的
  Best-Effort 队列，录制器只转换被 10 Hz 网格选中的图像。
- 将同步容差和延迟公开为 launch 参数：
  `recording_sync_tolerance_s`、`recording_sync_delay_s`。
- 为苹果 Full 添加目标独立工作区 `X=0.40..0.55`、`Y=-0.23..0.02 m`。
- 记录鼠标人工调试姿态到
  `config/mouse_manual_pose_snapshot_2026-09-10.yaml`。
- 鼠标抓取方向逻辑已按“沿鼠标长轴顺入、夹取两个侧面”调整；鼠标工作目前不是
  批量苹果采集的优先阻塞项。
- 桌面可视几何与 MoveIt 碰撞体已核对一致：顶面 `z=0`。
- 两路无人订阅的深度渲染产品（`/camera/camera/depth/...`、
  `/global_camera/depth/...`）改为默认关闭，只有显式传
  `run_z1_ros2_scene.py --depth-streams` 才创建；录制只消费三路 RGB。
- 录制动作因果性上限从 `1/fps`（10 fps 下 0.1 s，零余量）改为参数
  `action_age_s`，launch 参数 `recording_action_age_s`，默认 0.40 s；
  每帧新增 `action_age_s` 字段记录该帧实际使用的指令年龄。

### 验证状态

- Python/JSON 语法检查通过。
- 包内测试：`171 passed`（新增
  `test_command_pause_between_segments_keeps_the_frame`）。
- `z1_isaac_bridge` 使用 `PYTHONNOUSERSITE=1` 的 symlink build 成功。
- 相机同步与苹果 Full 范围修改尚未经过新的长时间连续采集实测。
- 深度流关闭与 `action_age_s` 改动已通过单测和静态断言，但尚未经过带物理的
  实际采集验证。

## 11. 已知失败模式

### 11.1 Full 预抓取规划失败（此前最大问题）

历史 Full 运行大量出现：

```text
candidate N: MoveIt error_code=99999
Unable to sample any valid states for goal tree
candidate N: joint delta ... > 2.400 rad
```

一次自动示教最多测试 3 个布局 × 8 个候选。旧 Full 全桌面范围没有考虑苹果固定
抓取姿态的实际可达域，导致大量系统性失败。最近一次旧范围运行日志累计出现
587 条候选 `99999`、74 条 joint-delta 拒绝；同时仍有 20 次双侧抓取成功，说明
并非所有失败都发生在同一阶段。

苹果 Full 范围已经缩小，但需要重新统计：

- 顶层尝试数
- 规划成功率
- 双侧抓取成功率
- 带载搬运成功率
- 最终保存率

### 11.2 抓住后搬运掉落

一些尝试能够完成双侧接触和带载试抬，但向蓝盒横移后左右接触都变为 `None`，
物体相对夹爪位置突变。历史日志中至少观察到 7 组明确的搬运后丢失。这是当前
缩小 Full 范围之后仍需处理的首要抓取问题。

不要通过附着物体或禁用物理来掩盖问题。应检查：

- 闭爪预载是否足够且不过度挤出苹果。
- 横移动作的速度/加速度和姿态变化。
- 带载轨迹是否触发接触丢失。
- 左右指接触冲量是否长期不对称。
- 搬运前后 stator-local 相对位姿。

### 11.3 相机时间同步失败

历史错误：

```text
写入帧失败: wrist|vla_wrist|global: no fresh image near <sim time>
```

此前相机平均只有约 10–11 Hz，但最大帧间隔达到 0.27–0.37 秒；旧录制容差仅
25 ms，任一路偶发卡顿都会让整轮不可保存。当前已提高 RGB 发布率并把容差/等待
改为 45 ms/0.5 s，仍需在没有其他 Isaac 进程、关闭 RViz的情况下实测。

若仍失败，先测三路实际频率和最大间隔，不要立即把容差扩大到 0.2 秒；过大容差
会把明显不同时刻的图像、状态与动作拼成同一训练帧。

2026-09-10 实测补充（单栈、`rviz:=false`、`full` 课程苹果目标）：

- 18 次保存失败，0 个 Episode：`global` 10 次、`action` 6 次、`wrist` 2 次。
- 失败模式与阶段强相关：`EXECUTING`（预抓取轨迹）阶段 10 次全是图像新鲜度，
  `GRASPING`（对准/闭爪，分段规划）阶段 6 次全是动作因果性。
- 平均速率不是问题（三路墙钟 24–29 Hz，按 RTF 折算仿真间隔约 22 ms），问题
  是卡顿：墙钟最大间隔 wrist 123 ms / vla_wrist 135 ms / global 175 ms，按
  RTF≈0.65 折算仿真时间 80–114 ms，是 45 ms 容差的 1.8–2.5 倍。
- 干净基线（`.recording_000000` 的 102 帧）里三路对齐误差中位 0 ms、最大
  33.3 ms，容差余量只有约 12 ms，因此任何渲染停顿都会立刻掉帧。
- 运行 25 分钟后 `global` 从 29.0 Hz / 最大 117 ms 退化到 24.4 Hz / 最大
  175 ms，RTF 从 0.70 降到 0.65；失败次数每 5 分钟递增 3→4→5→6。
- `/isaac_joint_commands` 虽平均约 102 Hz，但存在 0.492 s 墙钟 / 0.317 s 仿真
  时间空档，而旧上限 `1/fps` 只有 0.1 s。

对应的两处修改见 §10：关闭无人订阅的深度渲染产品（降低 `EXECUTING` 阶段的
渲染卡顿），以及 `action_age_s` 参数化并逐帧记录真实指令年龄（解除
`GRASPING` 阶段的零余量约束）。图像网格容差本身没有放宽，§11.3 上半部分的
警告仍然有效。
### 11.4 重复训练栈

曾误启动两套完整训练栈，导致：

- `/clock` 时间持续向后跳。
- TF buffer 被反复清空。
- 两个 RViz 以退出码 `-11` 崩溃。
- 第二个 GUI 因 8767 被占用退出。
- 同名节点、MoveIt 和控制命令发生竞争。

遇到大面积异常时，第一步必须检查进程和 `/clock` 发布者数量。

## 12. 建议下一步

1. 确认没有其他 Isaac Sim 和旧 Z1 栈。
2. 使用 `rviz:=false` 启动一套训练栈。
3. 先运行 10 次苹果 `easy`，验证相机同步修改和完整保存链路。
4. 再运行带 `--max-attempts` 的小规模苹果 `full`，例如目标 10、最多尝试 30。
5. 记录四段成功率：规划、抓取、搬运、保存。
6. 若 Full 规划率仍低，根据失败位置继续调整苹果目标工作区或加入离线可达性掩码；
   不要只增加布局重试次数。
7. 若抓取率高而保存率低，优先修复搬运掉落。
8. 连续采集稳定后再开始大规模 100 轮采集。
9. 为鼠标、瓶子等目标分别建立经过验证的 `target_workspaces_m`，不要共用全桌面
   矩形。
10. 更新 LeRobot 转换器默认输入/输出路径；当前转换器默认输入仍是旧的
    `z1_tabletop_raw_16x9`。在修改前必须显式传递：

```bash
conda run -n unifolm-vla python \
  /home/lp531/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py \
  --input /home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist \
  --output /home/lp531/smolvla_data/z1_tabletop_lerobot_dual_wrist
```

转换器会从 `dataset_info.json` 自动发现三路相机键，但不会覆盖已经存在的输出目录。

## 13. 推荐诊断命令

```bash
# 总状态
curl -s http://127.0.0.1:8767/api/status | python3 -m json.tool

# 关键进程
pgrep -af "continuous_training|z1_isaac.launch|run_z1_ros2_scene|move_group"

# ROS 同名节点与时钟发布者
ros2 node list | sort | uniq -c
ros2 topic info /clock --verbose

# 三路 RGB 发布频率
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /vla_wrist_camera/color/image_raw
ros2 topic hz /global_camera/color/image_raw

# 录制状态
ros2 topic echo /smolvla/recording_status

# 抓取综合状态
ros2 topic echo /smolvla/status

# 最近日志中的主要错误
rg "MoveIt error_code|joint delta|no fresh image|搬运后|task_success" \
  /home/lp531/.ros/log/*.log
```

## 14. 修改时必须保持的工程约束

- 不允许用虚拟附着、磁吸、物体跟随或关闭重力伪造抓取成功。
- 失败 Episode 不得保存为训练数据。
- 只有完整物理抓取、抬升、释放和稳定入盒才能保存。
- 规划必须保留桌面、蓝盒、机械臂、相机和其他物体碰撞检查。
- 不要无条件提高关节变化阈值或忽略 `MoveIt error_code=99999`。
- 不要同时运行两套同名 ROS/Isaac 训练栈。
- 修改统一几何/物理参数时优先改 `config/grasp_physics.json`，避免在多个脚本复制
  常数。
- 修改源码后运行 170 项测试，并用 `PYTHONNOUSERSITE=1` 重新构建。
- 用户工作区可能没有 Git 根仓库；不要假设可以通过 `git reset` 恢复文件。

## 15. 交接给新 Agent 的起始任务

新 Agent 应先阅读本文档，然后依次阅读：

1. `config/grasp_physics.json`
2. `z1_isaac_bridge/continuous_training.py`
3. `z1_isaac_bridge/grasp_gui_bridge.py` 中 `auto_demonstration()`
4. `z1_isaac_bridge/episode_recorder.py` 中采样与保存回调
5. `z1_isaac_bridge/grasp_assistant.py` 中候选生成、物理试抬和搬运逻辑
6. `/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py` 中随机化和相机图

接手后的第一项实测应是：在单一 Isaac、关闭 RViz的条件下，用新相机同步参数
运行少量 `easy`，确认能产生首个合法 `episode_000000`；随后才验证新的苹果 Full
范围和搬运成功率。

## 16. 2026-09-12 相机同步修复更新（优先于上述旧状态）

已通过默认/64 MiB 共享内存的独立进程对照试验证实：Fast DDS 默认
512 KiB 段不足以容纳 691200–921600 字节 RGB 图像，造成缺帧。
新增 `config/fastdds_camera.xml`，launch 自动设置
`FASTRTPS_DEFAULT_PROFILES_FILE`，可通过 `fastdds_profile` 覆盖。
保持 10 Hz、45 ms 容差、0.5 秒延迟，不复制图像或放宽保存校验。

已编译、重启单一 Isaac 栈，并验证环境和内存映射实际加载配置。
相关 14 项回归测试通过；完整回合 `episode_000003` 已成功保存
410 帧/41 秒，零丢帧，图像时间误差最大 2.139 微秒，无重复图像。
目前累计 3/100，恢复脚本以 `apple / easy / --episodes 98` 运行，
修复后进度 1/98。抓取定位、夹持和 MoveIt 失败仍会正常丢弃重试。
不要再按上一节“先创建 episode_000000”的旧状态处理数据。

完整证据、偶发订阅缺口的限制及复测命令见
`../docs/camera_sync_transport_fix_2026-09-12.md`。
本次循环日志：`/tmp/z1-continuous-sync-20260912.log`。
