# 从 Z1 URDF 到 Isaac Sim SmolVLA 训练环境

## 1. 总体链路

当前项目并不是把 URDF 直接交给 SmolVLA 训练，而是经过以下完整链路：

```text
Z1 Xacro/URDF
    ↓ 展开、解析网格路径
Isaac Sim Z1 Articulation USD
    ↓ 加入相机、驱动器和物理属性
SmolVLA 桌面任务 USD
    ↓ Isaac ROS 2 OmniGraph
ROS 2 / MoveIt / 抓取控制与物理验证
    ↓ 同步录制图像、关节状态和动作
原始 Episode 数据
    ↓ 转换
LeRobot 数据集
    ↓
SmolVLA 微调
```

### 工作方式标记

本文后续使用以下三种标记区分责任边界：

- **【人工】**：需要操作者测量、判断、选择或确认，脚本不能可靠代替；
- **【脚本】**：输入准备正确后，由程序自动完成，不应在 Isaac GUI 中逐项手工修改；
- **【人工触发 + 脚本执行】**：操作者运行命令或点击前端按钮，后续步骤由程序编排。

最重要的原则是：**人负责确定什么是正确的机器人、场景和示教结果；脚本负责重复地构建、运行、记录和转换。**

## 2. Z1 的原始机器人描述

机械臂的源文件是：

```text
/home/lp531/ros2_ws/src/z1_ros2/z1_description/urdf/z1.urdf.xacro
```

它定义了：

- `link00～link06` 六轴机械臂链；
- `joint1～joint6` 六个旋转关节；
- `gripperStator` 固定夹指；
- `gripperMover` 活动夹指；
- `jointGripper` 夹爪开合关节；
- 每个连杆的质量、惯量、关节限位、阻尼和摩擦；
- DAE 外观网格；
- STL 机械臂碰撞网格；
- 固定在 `gripperStator` 上的两套 D435 相机坐标链。

第一套 D435 使用真实机械臂的 ChArUco 手眼标定结果；第二套是为了改善 VLA 抓取区域视野增加的顶置腕部相机。

相关文件：

- 相机宏：`z1_description/urdf/_d435.urdf.xacro`
- ROS 控制接口：`z1_description/urdf/z1.ros2_control.xacro`
- 视觉网格：`z1_description/meshes/visual/`
- 碰撞网格：`z1_description/meshes/collision/`

因此，这份 Xacro 同时承担三个角色：

1. 给 `robot_state_publisher` 和 TF 使用；
2. 给 MoveIt 建立运动学模型；
3. 作为 Isaac Sim 机械臂 USD 的生成源。

## 3. 从 Xacro 生成普通 URDF

本阶段属于 **【人工触发 + 脚本执行】**：人工决定何时重建并执行命令，`xacro` 自动完成宏展开。正常情况下不需要手工编辑生成的 `.urdf`。

Isaac 的 URDF 导入器不能直接处理 ROS Xacro，因此首先需要将其展开：

```bash
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash

xacro \
  /home/lp531/ros2_ws/src/z1_ros2/z1_description/urdf/z1.urdf.xacro \
  sim_ignition:=false \
  with_gripper:=true \
  -o /home/lp531/isaac_assets/z1/generated/z1_with_d435.urdf
```

得到：

```text
/home/lp531/isaac_assets/z1/generated/z1_with_d435.urdf
```

这一步会将 Xacro 常量、条件分支、相机宏、夹爪宏、惯量与限位表达式全部展开成普通 XML URDF。

随后，场景构建脚本将：

```text
package://z1_description/meshes/...
```

替换成 Isaac 可以直接读取的绝对路径，并生成：

```text
/home/lp531/isaac_assets/z1/generated/z1_with_d435_resolved.urdf
```

对应实现位于：

```text
/home/lp531/isaac_assets/z1/scripts/build_smolvla_red_ball_scene.py
```

## 4. 将 URDF 导入 Isaac，生成机器人 USD

本阶段属于 **【脚本】**。机器人 USD 应由构建脚本重新生成，不建议在 Isaac GUI 中逐关节手工重复配置，因为下一次重建会覆盖这些临时修改。

构建脚本调用 Isaac Sim 的 `URDFParseAndImportFile`，完成以下转换：

```text
z1_with_d435_resolved.urdf
        ↓ Isaac URDF Importer
z1_with_d435.usd
```

生成的机器人资产是：

```text
/home/lp531/isaac_assets/z1/generated/z1_with_d435.usd
```

当前导入配置的关键点包括：

- `fix_base=True`：固定 Z1 底座；
- `merge_fixed_joints=False`：不合并固定关节，保留 D435 的 TF 坐标链；
- 使用 URDF 自带碰撞网格，不从视觉网格生成碰撞；
- 导入质量和惯量张量；
- 解析夹爪及相机固定关节；
- 将 Z1 创建为 PhysX articulation；
- 关闭机械臂内部自碰撞，由 MoveIt 和指定碰撞规则负责规划安全性。

导入后，脚本还会为七个可控关节写入 Isaac 驱动参数：

```text
joint1～joint6
jointGripper
```

每个关节分别配置：

- stiffness；
- damping；
- max force。

夹爪驱动力比大臂关节低，避免在仿真中产生不符合实际的无限夹持力。

## 5. 相机如何进入 Isaac 训练环境

本阶段同时包含人工标定和脚本建模：

- **【人工】** 安装真实 D435、采集 ChArUco 样本、检查图像视野，并判断标定结果是否合理；
- **【脚本】** 计算/读取标定参数、在 URDF 中维护相机坐标链，并在机器人 USD 下创建 Camera Prim；
- **【人工】** 对新增 VLA 相机的安装位置进行可见性和机械干涉复核；
- **【脚本】** 按配置生成相机内参、分辨率、裁剪范围和 ROS 2 发布节点。

URDF 中保留相机的物理安装坐标和 TF 链，真正负责渲染出图的 `UsdGeom.Camera` 由场景构建脚本创建在对应光学坐标系之下。

当前包含三路 RGB 观察：

| 观察源 | 分辨率 | 作用 |
|---|---:|---|
| 手眼标定腕部 D435 | 640×360 | 对应真实机械臂相机 |
| VLA 顶置腕部相机 | 640×360 | 增强抓取区域可见性 |
| 全局第三人称相机 | 640×480 | 观察机械臂和整个桌面 |

相机内参来自真实 D435i 标定值；全局相机使用仿真基线内参。

当前默认运行的双腕部相机场景是：

```text
/home/lp531/isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd
```

相机说明位于：

```text
/home/lp531/isaac_assets/z1/generated_dual_camera/README.md
```

Isaac 相机坐标约定与 ROS optical frame 不同，因此 Camera Prim 带有局部 180° X 轴旋转。该旋转只负责坐标约定转换，不改变真实手眼外参。

## 6. 从机器人 USD 组成完整训练场景

场景的几何与训练意图由人工确定，实际 USD 则由脚本生成：

- **【人工】** 选择桌面、蓝盒、训练物体、初始布局和任务语言；
- **【人工】** 测量或确认物体质量、尺寸、摩擦需求及合理抓取区域；
- **【脚本】** 创建桌面、蓝盒、成功区域、灯光、相机、RigidBody 和碰撞代理；
- **【脚本】** 写入语义标签、随机化范围及场景元数据；
- **【人工】** 在 Isaac/RViz 中检查外观与 collider 是否明显不一致。

机器人 USD 只是机械臂资产。场景构建脚本进一步创建任务环境：

```text
/World
├── Z1
├── PhysicsScene
├── Task
│   ├── WorkSurface
│   ├── Objects
│   │   ├── Cup
│   │   ├── Apple
│   │   ├── Peach
│   │   ├── Mouse
│   │   └── BeverageBottle
│   └── Box
│       ├── Bottom
│       ├── WallLeft
│       ├── WallRight
│       ├── WallFront
│       ├── WallBack
│       └── SuccessVolume
├── GlobalCamera
└── Lights
```

早期兼容场景文件名仍然是：

```text
/home/lp531/isaac_assets/z1/generated/smolvla_red_ball_into_box.usd
```

虽然文件名带有 `red_ball`，实际内容已经是五物体 SmolVLA 桌面抓取场景。

### 6.1 训练物体

训练物体采用两层结构：

- 高质量扫描资产负责外观；
- 简化的物理代理负责稳定碰撞。

当前主要碰撞代理为：

- 苹果、桃子：椭球碰撞体；
- 饮料瓶：半径约 `24.2 mm`、高度 `157 mm` 的圆柱碰撞体；
- 鼠标：椭球代理；
- 杯子：圆柱代理。

因此，在 RViz 中看到的是圆柱或椭球，而不是扫描资产的复杂外观网格。MoveIt 和 PhysX 使用简化代理，是为了获得更稳定、可预测的碰撞与规划结果。

### 6.2 物理与任务属性

场景同时配置：

- 重力 `9.81 m/s²`；
- TGS 物理解算器；
- CCD 连续碰撞检测；
- 物体质量、摩擦和阻尼；
- 桌面与夹爪物理材料；
- 蓝盒成功区域；
- 物体语义标签；
- easy、medium、full 三档场景随机化。

## 7. Isaac 与 ROS 2 的连接

本阶段的通信图与节点实现属于 **【脚本】**；启动、选择 GUI/headless 模式及处理运行异常属于 **【人工触发 + 脚本执行】**。

ROS 2 总启动入口是：

```text
/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/launch/z1_isaac.launch.py
```

Isaac 场景运行脚本是：

```text
/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py
```

Isaac 内部创建 ROS 2 OmniGraph，完成以下通信。

### Isaac → ROS 2

```text
/joint_states
/clock
相机图像与 camera_info
物体 TF
接触报告
场景状态
```

### ROS 2 → Isaac

```text
/isaac_joint_commands
场景复位
场景随机化
物体冻结/解冻
抓取探测命令
```

Isaac Sim 使用 Python 3.11，ROS Humble 工作空间使用 Python 3.10。项目没有直接混合两个 Python 环境，而是：

- Isaac 使用自带的 Humble ROS 2 bridge；
- ROS 工作空间运行标准 Humble 节点；
- 两边通过 DDS 消息通信。

这样可以避免 Python ABI 和 ROS 动态库冲突。

## 8. MoveIt 如何控制 Isaac 中的 Z1

MoveIt 配置、控制桥和轨迹转发属于 **【脚本】**。人工主要负责检查模型是否一致，以及在异常运动时使用强制停止。

MoveIt 不直接操作 Isaac articulation，中间使用：

```text
/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/
z1_isaac_bridge/trajectory_bridge.py
```

控制链路如下：

```text
MoveIt
  ↓ FollowJointTrajectory
trajectory_bridge
  ↓ JointState 位置命令
/isaac_joint_commands
  ↓
Isaac Articulation Controller
  ↓
Z1 PhysX 关节
```

反馈链路如下：

```text
Isaac Z1
  ↓
/joint_states
  ↓
trajectory_bridge + robot_state_publisher + MoveIt
```

桥接器分别提供：

- 六轴机械臂轨迹控制器；
- 单独的夹爪轨迹控制器；
- 强制停止并保持当前位置的服务；
- 机械臂与夹爪关节限位检查。

因此，RViz、MoveIt 和抓取助手看到的是标准 ROS 2 机械臂接口，真正执行动力学的是 Isaac Sim。

## 9. 抓取训练环境增加的功能

抓取动作主要是 **【人工触发 + 脚本执行】**：操作者选择目标、随机化等级并点击一键示教，后端自动完成规划、接近、闭环微调、闭爪、抬升、运输、释放与结果判定。

当前系统已经不只是让机械臂运动，还增加了完整的自动抓取、验证、放置和录制状态机。

主要组件包括：

| 组件 | 作用 |
|---|---|
| `grasp_assistant.py` | 生成预抓取、闭环对中、闭爪、抬升、运输和放置动作 |
| `grasp_evaluator.py` | 判断是否真实抓住、抬起并成功放入盒子 |
| `grasp_physics.json` | 配置物体尺寸、抓取区间、摩擦、接触门槛和抬升标准 |
| `planning_scene_sync.py` | 将桌面和蓝盒同步到 MoveIt |
| `object_collision_sync.py` | 将动态物体位置同步到 MoveIt |
| `episode_reset.py` | 安全停止、解冻、复位和重新随机化 |
| `grasp_gui_bridge.py` | 为前端提供自动示教、状态和强制停止接口 |

现在一次抓取需要依次满足：

1. 物体进入抓取几何区间；
2. 左右夹指真实接触目标；
3. 接触保持足够帧数；
4. 接触点位于物体允许抓取区域；
5. 左右接触冲量比例合理；
6. 物体离开桌面；
7. 抬升过程中没有失去双指夹持；
8. 物体放入蓝盒并稳定；
9. 夹爪完成释放。

只有满足这些物理与任务条件，episode 才允许保存为有效训练数据。

## 10. 如何生成 SmolVLA 原始训练数据

录制本身属于 **【脚本】**，但是否把一次执行视为可信示教仍需要 **【人工】** 抽样复核。程序门禁可以排除已知物理失败，却不能完全替代对视频中穿模、错误接触和异常捷径的视觉检查。

数据录制节点是：

```text
/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/
z1_isaac_bridge/episode_recorder.py
```

默认以 10 Hz 同步记录：

```text
observation.images.wrist
observation.images.vla_wrist
observation.images.global
observation.state
action
语言任务
随机化信息
物理抓取与任务成功证据
```

其中：

- `observation.state`：七个关节的真实位置；
- `action`：发送给 Isaac 的七个目标关节位置；
- 图像：两路腕部 RGB 和一路全局 RGB；
- task：例如 `Physically grasp the bottle, lift it clear, and place it in the box.`。

原始数据默认保存到：

```text
/home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist
```

当前原始数据格式为：

```text
z1_smolvla_raw_v1
```

每个 episode 包含：

```text
episode_start.json
frames.jsonl
episode_summary.json
images/
```

录制过程首先写入 `.recording_NNNNNN` 临时目录。只有通过保存门禁后，才会原子重命名为 `episode_NNNNNN`。失败抓取会被丢弃，避免直接污染正式训练集。

## 11. 转换为 LeRobot/SmolVLA 数据格式

格式转换属于 **【人工触发 + 脚本执行】**：人工选择输入数据集、输出目录、相机 schema 和是否包含失败样本，转换器自动生成 Parquet、视频、任务表及统计量。

转换脚本是：

```text
/home/lp531/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py
```

它完成以下转换：

```text
原始 JPEG + frames.jsonl
        ↓
LeRobot v3
├── Parquet 状态与动作
├── 多路 MP4 或图像
├── tasks
├── episode metadata
└── statistics
```

转换器默认只选择 `task_success=true` 的 episode，并在结束时重新加载生成的数据集，验证 episode 数、帧数和 FPS。

示例转换命令：

```bash
conda run -n unifolm-vla python \
  /home/lp531/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py \
  --input /home/lp531/smolvla_data/z1_tabletop_raw_dual_wrist \
  --output /home/lp531/smolvla_data/z1_tabletop_lerobot_dual_wrist \
  --repo-id lp531/z1_smolvla_tabletop
```

当前已经存在一个转换后的实验数据集：

```text
/home/lp531/smolvla_data/z1_tabletop_lerobot_16x9_episode_000003
```

默认三相机数据目录仍主要处于采集和物理有效性筛选阶段。

## 12. 人工操作与脚本自动化边界

### 12.1 按阶段划分

| 阶段 | 人工需要完成 | 脚本自动完成 | 当前主要入口 |
|---|---|---|---|
| 机械模型准备 | 确认连杆尺寸、质量、惯量、关节轴、限位和夹爪结构 | Xacro 宏展开、生成普通 URDF | `xacro` + `z1.urdf.xacro` |
| 网格与碰撞 | 选择正确 DAE/STL，检查比例、原点和碰撞覆盖范围 | 解析 `package://` 路径并验证网格存在 | `build_smolvla_red_ball_scene.py` |
| 相机标定 | 安装 D435、采集标定板样本、复核外参与视野 | 读取参数并生成相机 TF、Camera Prim 和发布配置 | 手眼标定脚本 + 场景构建脚本 |
| 机器人 USD | 决定是否固定底座、保留固定关节及驱动力范围 | URDF 导入、设置 articulation/drive、清理无效引用 | `build_smolvla_red_ball_scene.py` |
| 任务场景 | 选择物体、布局、蓝盒位置、任务目标和随机化范围 | 创建 USD 场景、RigidBody、collider、材质、灯光及语义 | `build_smolvla_red_ball_scene.py` |
| 扫描资产 | 选择来源并核对授权、真实尺寸和外观方向 | 规范化 USDZ/USD 包装并生成资产清单 | `prepare_scanned_assets.py` |
| 场景验收 | 查看模型比例、相机画面、碰撞代理和关节运动 | 自动检查 Prim、相机输出和关键资产是否存在 | `smoke_test_smolvla_scene.py` |
| 系统启动 | 选择 GUI/headless、目标物体和数据目录 | 启动 Isaac、MoveIt、RViz、桥接器、评估器与录制器 | `z1_isaac.launch.py` |
| 场景随机化 | 选择 easy/medium/full 或 seed，观察布局是否合理 | 采样无重叠物体位置、应用姿态并验证场景 | 前端或 `/smolvla/randomize_*` |
| 自动示教 | 选择目标并决定是否开始/强停/重试 | 规划、接近、微调、闭爪、抬升、放置和状态编排 | 8767 前端 + `grasp_assistant.py` |
| 抓取参数校准 | 根据实际接触点和失败视频决定高度、倾角、容差 | 使用新参数闭环执行并输出量化失败原因 | `grasp_physics.json` + 抓取节点 |
| Episode 录制 | 决定开始、保存、丢弃并抽查视觉真实性 | 同步多相机、状态、动作和成功证据，原子保存 | `episode_recorder.py` |
| 数据清洗 | 隔离穿模、假抓取、错误任务和不可信历史数据 | 校验 schema、帧连续性、最低帧数和成功门禁 | 录制门禁 + 审核脚本/人工复核 |
| LeRobot 转换 | 选择经过审核的数据、输出路径和相机集合 | 生成 Parquet、MP4、任务表、统计量并重新加载验证 | `export_z1_lerobot_dataset.py` |
| SmolVLA 微调 | 选择基础模型、数据版本、训练超参数和验收指标 | 批处理、反向传播、checkpoint 与训练日志 | LeRobot `lerobot-train`（正式配置仍待确定） |
| 策略评估 | 设定未见 seed、成功率门槛并人工检查异常行为 | 批量运行 rollout、统计物理抓取和放置成功率 | 评估流程尚需在稳定数据后固化 |

### 12.2 只需要首次或硬件变化后人工完成的工作

以下工作不是每次启动都要做：

1. 核对 Z1 连杆、惯量、关节轴和关节限位；
2. 测量夹爪开口、指尖几何和真实可夹持宽度；
3. 安装 D435 并完成手眼标定；
4. 确认相机不会与夹爪、物体或桌面发生机械干涉；
5. 选择扫描资产并核对许可证、单位、朝向和尺度；
6. 为新物体确定质量、摩擦、碰撞代理和有效抓取区域；
7. 确定任务定义、语言指令、蓝盒成功区域和随机化边界。

当 URDF、相机安装位姿、网格或物理配置发生变化时，需要重新执行对应检查。

### 12.3 每次修改模型后由脚本重建的工作

只要修改了以下任一内容，就应重新运行构建流程：

- `z1.urdf.xacro` 或其 include 文件；
- DAE/STL 网格；
- D435 安装外参或相机链；
- 机器人关节、夹爪结构或惯量；
- 训练物体资产、碰撞形状或场景布局生成逻辑。

重建时脚本依次完成：

1. 展开 Xacro；
2. 解析 ROS 网格 URI；
3. 导入机器人 USD；
4. 设置关节 drive；
5. 创建相机；
6. 生成任务场景；
7. 写入元数据；
8. 运行 smoke test。

生成文件不应作为主要编辑源。需要修改时，应修改 Xacro、JSON 配置或构建脚本，然后重新生成。

### 12.4 每次启动系统时人工完成的工作

1. 启动完整 ROS 2/Isaac 系统；
2. 确认 Isaac、MoveIt、RViz 和 8767 前端正常连接；
3. 检查 `/clock`、`/joint_states` 和三路 RGB 图像是否更新；
4. 确认机械臂处于安全初始位，夹爪没有残留控制目标；
5. 必要时执行“重置 Episode”；
6. 选择目标物体和随机化等级；
7. 再启动自动示教或试抓取。

启动以后，ROS 节点、话题、服务、控制器和相机发布由 launch 文件自动创建，不需要逐项手动打开。

### 12.5 每次试抓取中的人工与自动步骤

人工操作：

1. 在前端选择目标物体；
2. 选择 easy、medium 或 full；
3. 点击一键示教；
4. 若出现危险运动、死循环或明显穿模，点击强制停止；
5. 观察最终接触、抬升和放置是否符合实际。

脚本自动执行：

1. 场景复位与随机化；
2. MoveIt 预抓取规划；
3. 移动到安全对中位；
4. 使用 Isaac 实测 TF 进行闭环对中；
5. 闭爪并检查持续双指接触；
6. 检查接触点、接触冲量与物体专用几何策略；
7. 分段抬升并验证物体离开桌面；
8. 运输到蓝盒并释放；
9. 判断任务是否成功；
10. 成功则保存 episode，失败则丢弃失败录制并保留诊断信息。

### 12.6 哪些结果不能只相信脚本

以下情况即使程序返回成功，也需要人工视频复核或抽样检查：

- 物体穿透夹指或桌面；
- 使用了不真实的冻结、附着或约束捷径；
- collider 与视觉外观差异过大；
- 相机画面中目标长期被机械臂遮挡；
- 物体虽进入蓝盒，但动作明显依赖偶然碰撞；
- 关节动作存在剧烈跳变或不适合迁移到真机；
- 任务语言与实际被操作物体不一致。

因此，`task_success=true` 是自动门禁，不等同于最终数据质量认证。正式训练前仍应进行人工抽样审核，并将不可信 episode 移入隔离目录。

### 12.7 SmolVLA 训练阶段当前的责任边界

当前仓库已经具备原始数据录制和 LeRobot 转换脚本，但正式 SmolVLA 训练配置尚未固化为本项目的一键脚本。因此：

- **【脚本已完成】** 原始 episode 录制、成功门禁、LeRobot 格式转换；
- **【人工尚需决定】** 使用哪一版数据集、基础 SmolVLA checkpoint、训练步数、batch size、GPU 配置和验证划分；
- **【后续应脚本化】** 固定训练命令、保存配置快照、定期评估 checkpoint，并在未见随机 seed 上统计真实物理成功率。

在饮料瓶等物体尚未稳定抓取之前，不建议直接把所有现有 episode 合并后开始多物体正式训练。

## 13. 当前项目进度

目前已经完成：

- Z1 URDF/Xacro 机械模型；
- 夹爪和双 D435 相机链；
- URDF 到 Isaac USD 的自动构建；
- 多物体桌面和蓝盒训练场景；
- ROS 2、MoveIt 与 Isaac 双向控制；
- 场景随机化；
- 多相机同步录制；
- LeRobot 数据转换工具；
- 苹果和桃子的部分成功示教；
- 饮料瓶接近、闭爪和物理验证框架。

仍在完成：

- 饮料瓶可靠抓取姿态；
- 各物体在 easy、medium、full 随机位置下的成功率；
- 清理和隔离历史上“状态显示成功但实际没有物理夹住”的无效 episode；
- 扩充跨物体、跨位置的高质量三相机训练数据。

因此，当前准确阶段是：

> SmolVLA 仿真训练环境和数据管线已经构建完成；项目现在主要处于可靠示教数据采集、物理抓取校准和数据质量筛选阶段，尚未进入成熟的大规模多物体 SmolVLA 微调阶段。

饮料瓶等物体稳定通过真实双指接触、抬升和放置后，才适合开始正式的多物体 SmolVLA 微调。

## 14. 常用启动与重建命令

### 14.1 启动完整系统（人工触发，launch 自动编排）

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

### 14.2 修改 Xacro 后重建 Isaac 场景（人工触发，脚本生成）

```bash
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash

xacro \
  /home/lp531/ros2_ws/src/z1_ros2/z1_description/urdf/z1.urdf.xacro \
  sim_ignition:=false \
  with_gripper:=true \
  -o /home/lp531/isaac_assets/z1/generated/z1_with_d435.urdf

/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/prepare_scanned_assets.py

/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/build_smolvla_red_ball_scene.py

/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/smoke_test_smolvla_scene.py
```

## 15. 关键文件索引

| 文件 | 用途 |
|---|---|
| `/home/lp531/ros2_ws/src/z1_ros2/z1_description/urdf/z1.urdf.xacro` | Z1 机器人、夹爪和相机源描述 |
| `/home/lp531/isaac_assets/z1/scripts/build_smolvla_red_ball_scene.py` | URDF 导入、机器人 USD 与任务场景构建 |
| `/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py` | 运行 Isaac 场景、ROS 2 OmniGraph、物理接触和随机化 |
| `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/launch/z1_isaac.launch.py` | 完整系统启动入口 |
| `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/trajectory_bridge.py` | MoveIt 轨迹到 Isaac 关节命令桥接 |
| `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/grasp_assistant.py` | 自动抓取和放置状态机 |
| `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/grasp_evaluator.py` | 物理抓取与任务成功判定 |
| `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json` | 物理抓取参数与物体配置 |
| `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/episode_recorder.py` | 同步录制原始 episode |
| `/home/lp531/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py` | 转换为 LeRobot 数据集 |
