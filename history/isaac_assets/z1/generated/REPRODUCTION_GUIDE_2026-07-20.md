# Z1 URDF → Isaac Sim 场景 → ROS 2 / MoveIt 复现记录

记录日期：2026-07-20  
工作目录：`/home/lp531`  
目标：从 ROS 2 的 Unitree Z1 Xacro 生成 Isaac Sim 机器人和多物体训练场景，
并通过标准 ROS 2 `FollowJointTrajectory` 接口连接 MoveIt 与 RViz。

> 这是当前可工作的完整复现入口。历史调查过程保留在
> `WORK_LOG_2026-07-17.md` 与 `WORK_LOG_2026-07-20.md`。

## 1. 当前已实现的结果

- Z1 六轴机械臂及单自由度夹爪；
- 真实安装的 Intel RealSense D435/D435i TF 链；
- 基于真机 ChArUco 标定的腕部相机外参；
- 基于已连接相机工厂标定的 RGB、Depth 内参；
- Isaac Sim RGB、Depth、关节状态与仿真时钟发布；
- MoveIt/RViz 规划和执行；
- ROS 2 标准机械臂、夹爪轨迹 Action；
- 杯子、苹果、桃子、鼠标、饮料瓶五类高质量纹理资产；
- 蓝色目标盒、工作台、碰撞体、质量、摩擦与任务元数据；
- 红球已从训练场景中删除；
- RViz 与 Isaac Sim 轨迹使用同一 `/clock`，最终六轴最大测试误差
  `0.0047 rad`，约 `0.27°`。
- Isaac 动态物体真值发布与 ROS 2 抓取/放置成功判定节点。
- GUI/ROS 一键完整复位 Episode：停止轨迹、丢弃失败录制、机械臂回 home、
  夹爪打开、五个物体归位，并重置判定与 MoveIt 动态碰撞。

最终场景仍叫 `smolvla_red_ball_into_box.usd`，这是为了兼容已有启动命令；
文件内容已经是无红球的多物体场景。

## 2. 已验证环境

```text
OS: Ubuntu 22.04.5 LTS
ROS 2: Humble
系统 Python: 3.10.12
Isaac Sim: 5.1.0
GPU: NVIDIA GeForce RTX 4090
NVIDIA Driver: 580.159.03
ROS RMW: rmw_fastrtps_cpp
```

Isaac Sim 安装位置：

```text
/home/lp531/isaac-sim
```

ROS 2 工作空间：

```text
/home/lp531/ros2_ws
```

资产工作区：

```text
/home/lp531/isaac_assets/z1
```

## 3. 必须备份的内容

### 3.1 ROS 2 源码

```text
/home/lp531/ros2_ws/src/z1_ros2
```

上游仓库：

```text
https://github.com/idra-lab/z1_ros2.git
```

当前上游基线提交：

```text
d2462f735bd20cfd45c22692f9ddfbc3c7bb1cd5
```

注意：当前可运行版本包含未提交的本地修改和新增文件。只重新克隆上述提交
不能得到当前功能。至少必须保存：

```text
z1_description/urdf/z1.urdf.xacro
z1_description/urdf/_d435.urdf.xacro
z1_description/meshes/visual/d435.dae
z1_description/urdf/z1.ros2_control.xacro
z1_moveit/config/joint_limits.yaml
z1_moveit/config/moveit_controllers.yaml
z1_moveit/config/z1_description.srdf
z1_moveit/launch/move_group.launch.py
z1_moveit/launch/moveit_rviz.launch.py
z1_moveit/launch/z1_moveit.launch.py
z1_examples/z1_examples/camera_info_fix.py
z1_isaac_bridge/
```

最可靠的方法是直接归档整个 `src/z1_ros2`，而不是只保存 Git 提交号。

### 3.2 Isaac 场景源文件和构建脚本

```text
/home/lp531/isaac_assets/z1/scripts/build_smolvla_red_ball_scene.py
/home/lp531/isaac_assets/z1/scripts/prepare_scanned_assets.py
/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py
/home/lp531/isaac_assets/z1/scripts/smoke_test_smolvla_scene.py
```

### 3.3 不能由代码自动恢复的外部模型

必须保存整个目录：

```text
/home/lp531/isaac_assets/z1/scanned_objects/source
```

其中当前使用：

```text
cup.usdz
apple.usdz
peach.usdz
mouse.usdz
beverage_bottle.usdz
```

模型来源、作者、许可证、原始链接和 SHA-256 位于：

```text
/home/lp531/isaac_assets/z1/scanned_objects/manifest.json
/home/lp531/isaac_assets/z1/scanned_objects/ATTRIBUTION.md
/home/lp531/isaac_assets/z1/scanned_objects/UPSTREAM_CREDITS.csv
```

这些文件来自 `barszot/3d-models-for-isaac-sim-dataset` 收录的 CC
Attribution 模型。重新分发时必须同时保留署名文件。

### 3.4 相机标定

实际采用的手眼标定：

```text
/home/lp531/ros2_ws/z1_handeye_charuco_gripper_to_camera.yaml
```

不要误用旧文件：

```text
/home/lp531/isaac_assets/z1/hand_eye_calibration.yaml
```

后者记录的是早期虚拟 `WristCamera` 参数，不是当前 D435 真机外参。

## 4. 数据流和职责

```text
z1.urdf.xacro
    │ xacro 展开
    ▼
z1_with_d435.urdf
    │ Isaac URDF Importer
    ▼
z1_with_d435.usd
    │ 加入相机、关节驱动、工作台、蓝盒和五类物体
    ▼
smolvla_red_ball_into_box.usd
    │ run_z1_ros2_scene.py 创建 OmniGraph
    ├── /joint_states ───────────────► ROS 2 / robot_state_publisher / RViz
    ├── /clock ──────────────────────► MoveIt、RViz、轨迹桥
    ├── RGB/Depth 图像和 CameraInfo ─► ROS 2
    └── /isaac_joint_commands ◄────── trajectory_bridge.py
                                      ▲
                                      │ FollowJointTrajectory
                                  MoveIt / RViz
```

`trajectory_bridge.py` 对外模拟两个 ros2_control 控制器 Action，Isaac
不需要加载真实 Z1 硬件插件，也不需要 Gazebo。

失败后无需重启终端或 Isaac Sim，在 8767 GUI 点击红色“一键重置 Episode”，
或执行：

```bash
ros2 service call /smolvla/reset_episode std_srvs/srv/Trigger "{}"
```

完成后可直接选择目标并开始下一次录制。该服务会自动丢弃尚未保存的当前
episode；已保存的 `episode_NNNNNN` 不受影响。

## 5. 从零构建 ROS 2 工作空间

安装 ROS 2 Humble、MoveIt、Xacro、RealSense 描述包以及工作空间依赖后：

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash

rosdep install \
  --from-paths src \
  --ignore-src \
  --rosdistro humble \
  -r -y

colcon build --symlink-install
source install/setup.bash
```

只重新编译本次直接相关包：

```bash
colcon build --symlink-install \
  --packages-select \
  z1_description \
  z1_examples \
  z1_moveit \
  z1_isaac_bridge

source install/setup.bash
```

## 6. 从 Xacro 生成 Isaac 使用的 URDF

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

xacro \
  /home/lp531/ros2_ws/src/z1_ros2/z1_description/urdf/z1.urdf.xacro \
  sim_ignition:=false \
  with_gripper:=true \
  -o /home/lp531/isaac_assets/z1/generated/z1_with_d435.urdf
```

这里必须使用 `sim_ignition:=false`，避免把 Gazebo/ros2_control 插件写进
Isaac 导入用 URDF。运行 ROS 节点时启动文件会另外用
`sim_ignition:=true` 生成 `robot_description`。

D435 固定安装在：

```text
parent: gripperStator
child: camera_bottom_screw_frame
```

底部螺孔安装位姿：

```text
translation xyz m:
[0.0900930511, -0.0009311400, -0.0436741355]

rotation rpy rad:
[0.04037701, 0.03561724, 0.01525814]
```

最终 `gripperStator -> camera_color_optical_frame`：

```text
translation xyz m:
[0.1006887452, 0.0312032005, -0.0302586435]

quaternion xyzw:
[-0.5023940423, 0.5151748053, -0.4771697725, 0.5044840478]
```

## 7. 准备高质量物体资产

确认五个 USDZ 已放入：

```text
/home/lp531/isaac_assets/z1/scanned_objects/source
```

然后运行：

```bash
/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/prepare_scanned_assets.py
```

该脚本会：

- 保留高质量纹理和几何；
- 统一尺寸、原点与地面高度；
- 输出本地 USD wrapper；
- 去除源资产内嵌 Physics API；
- 生成尺寸、哈希、碰撞体与夹持兼容性清单；
- 拒绝超过 Z1 保守夹持宽度 `55 mm` 的模型。

当前夹持宽度：

| 物体 | 规划夹持宽度 |
| --- | ---: |
| 杯子 | 50.4 mm |
| 苹果 | 50.0 mm |
| 桃子 | 52.0 mm |
| 鼠标 | 48.0 mm |
| 饮料瓶 | 48.4 mm |

原咖啡杯由于明显超过夹爪能力，被移至：

```text
scanned_objects/retired/cup_mug_ungraspable.usdz
```

## 8. 构建 Isaac 机器人与任务场景

```bash
/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/build_smolvla_red_ball_scene.py
```

主要输出：

```text
generated/z1_with_d435_resolved.urdf
generated/z1_with_d435.usd
generated/configuration/
generated/smolvla_red_ball_into_box.usd
generated/smolvla_red_ball_into_box.json
```

场景使用米、Z-up、重力 `9.81 m/s²`、TGS solver、CCD 和稳定化。

初始布局（相对 Z1 `link00`，+X 向前，+Z 向上）：

| 物体 | XYZ m | Yaw |
| --- | --- | ---: |
| 杯子 | `(0.62, -0.08, 0.0)` | -20° |
| 苹果 | `(0.44, -0.18, 0.0)` | 12° |
| 桃子 | `(0.54, -0.16, 0.0)` | -8° |
| 鼠标 | `(0.49, -0.02, 0.0)` | 18° |
| 饮料瓶 | `(0.62, 0.08, 0.0)` | 5° |
| 蓝盒中心 | `(0.32, 0.14, 0.0)` | 0° |

工作台尺寸 `0.90 × 0.70 × 0.05 m`。蓝盒外尺寸 `0.16 m`，墙高
`0.10 m`，顶部开放。几何成功区域：

```text
/World/Task/Box/SuccessVolume
```

训练物体 Prim：

```text
/World/Task/Objects/Cup
/World/Task/Objects/Apple
/World/Task/Objects/Peach
/World/Task/Objects/Mouse
/World/Task/Objects/BeverageBottle
```

红球 `/World/Task/RedBall` 不应存在。

## 9. 相机参数

分辨率：

```text
640 × 480
```

RGB：

```text
fx=602.1416015625
fy=601.827819824219
cx=326.930267333984
cy=251.521286010742
```

Depth：

```text
fx=387.84747314453125
fy=387.84747314453125
cx=319.447265625
cy=238.202484130859
```

裁剪距离：

```text
0.05–5.0 m
```

Isaac renderer 会把自己推导的矩阵发布到 `camera_info_raw`。启动文件中的
`camera_info_fix.py` 保留时间戳、frame、畸变和图像尺寸，使用真机矩阵覆盖
`K/P`，再发布到标准 `camera_info`。

Isaac 的 `UsdGeom.Camera` 额外有局部 X 轴 180° 旋转，只用于把 ROS optical
frame（+Z 向前、+X 向右、+Y 向下）转换为 Isaac camera（局部 -Z 向前）；
这不是第二份物理外参。

## 10. ROS 2 / Isaac 通信实现

Isaac 运行脚本：

```text
/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py
```

它在运行时创建：

- ROS 2 Context；
- Playback Tick；
- 仿真 `/clock`；
- `ROS2PublishJointState`；
- `ROS2SubscribeJointState`；
- `IsaacArticulationController`；
- RGB、Depth Render Product 和发布节点。

ROS 2 轨迹桥：

```text
/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge
```

Action：

```text
/joint_trajectory_controller/follow_joint_trajectory
/gripper_trajectory_controller/follow_joint_trajectory
```

Topic：

```text
/joint_states
/isaac_joint_commands
/clock
/camera/camera/color/image_raw
/camera/camera/color/camera_info_raw
/camera/camera/color/camera_info
/camera/camera/depth/image_rect_raw
/camera/camera/depth/camera_info_raw
/camera/camera/depth/camera_info
/global_camera/color/image_raw
/global_camera/color/camera_info
/global_camera/depth/image_rect_raw
/global_camera/depth/camera_info
/tf
/tf_static
/isaac/object_transforms
/smolvla/target_object
/smolvla/grasp_success
/smolvla/grasped_object
/smolvla/task_phase
/smolvla/task_success
/smolvla/status
```

### 固定全局 RGB-D 相机

一键启动脚本会在旧版 USD 场景缺少相机时自动创建：

```text
Prim: /World/GlobalCamera/global_camera_optical_frame/GlobalCamera
frame: global_camera_optical_frame
position in world: [0.72, -0.72, 0.72] m
look-at point: [0.32, 0.02, 0.12] m
resolution: 640 x 480
fx = fy = 554.2562584, cx = 320.0, cy = 240.0
```

RGB 与深度对齐并共用该 optical frame。`z1_isaac.launch.py` 同时发布固定
`world -> global_camera_optical_frame` TF。上述参数是仿真采集基线；真机确定
相机型号和安装位置后，必须以实际标定的内参、外参替换。

检查：

```bash
ros2 topic hz /global_camera/color/image_raw
ros2 topic echo /global_camera/color/camera_info --once
ros2 run tf2_ros tf2_echo world global_camera_optical_frame
```

关节顺序：

```text
joint1, joint2, joint3, joint4, joint5, joint6, jointGripper
```

轨迹桥关键参数：

```text
command_rate: 100 Hz
goal_tolerance: 0.005 rad
goal_time_tolerance: 3.0 s
joint_limit_tolerance: 0.002 rad
```

插值、反馈和收敛判断使用 ROS 仿真时间 `/clock`。等待第一帧状态仍使用墙上
单调时间，以防仿真尚未启动时等待永不结束。

MoveIt 执行参数：

```text
allowed_execution_duration_scaling: 2.0
allowed_goal_duration_margin: 3.0 s
allowed_start_tolerance: 0.01 rad
```

Isaac 关节位置驱动增益固化在构建脚本中，并在运行时再次设置：

| 关节 | stiffness | damping | max effort |
| --- | ---: | ---: | ---: |
| joint1 | 100000 | 1000 | 1000 |
| joint2 | 100000 | 1000 | 2000 |
| joint3 | 150000 | 1200 | 1600 |
| joint4 | 150000 | 1200 | 1000 |
| joint5 | 50000 | 500 | 600 |
| joint6 | 20000 | 200 | 300 |
| jointGripper | 2000 | 100 | 80 |

这些参数用于仿真位置轨迹一致性，不代表真实 Z1 电机力矩控制模型。

### 抓取与放置判定

`grasp_evaluator` 默认随一键 launch 启动。Isaac 通过专用 TFMessage topic
`/isaac/object_transforms` 发布五个动态物体在 `world` 下的真值位姿。

抓取成功同时要求：

- `jointGripper <= -0.12 rad`；
- 物体相对初始桌面高度抬升至少 `0.025 m`；
- 物体到 `gripperStator` 距离不超过 `0.20 m`；
- 物体相对夹爪的位置变化不超过 `0.008 m`；
- 相对旋转变化不超过 `8°`；
- 上述相对稳定状态持续至少 `0.40 s`。

放置成功要求此前已经确认抓取，然后目标物体进入蓝盒
`SuccessVolume`、夹爪释放、物体速度不超过 `0.03 m/s`，并稳定
`0.50 s`。这是仿真运动学真值标签，不是假装存在的真机触觉信号。

选择目标：

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py target_object:=apple

ros2 topic pub --once \
  /smolvla/target_object \
  std_msgs/msg/String \
  "{data: beverage_bottle}"
```

重置判定器：

```bash
ros2 service call \
  /smolvla/reset_evaluator \
  std_srvs/srv/Trigger \
  "{}"
```

### 判定 GUI

一键 launch 默认同时启动本机网页桥。启动完成后打开：

```text
http://127.0.0.1:8767
```

该页面可查看完整判定条件、切换五种训练目标并复位当前 episode。目标切换与
复位只改变判定状态，不会移动机械臂或物体；机械臂仍在 RViz 中手动遥操。

可选参数：

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py task_gui_port:=8770
ros2 launch z1_isaac_bridge z1_isaac.launch.py task_gui:=false
```

旧的 8765 遥操页面也已加入相同判定区，默认代理
`http://127.0.0.1:8767`。当前联动场景不要为了查看该区域额外启动旧遥操
backend，因为旧 backend 会自行再创建一套 Isaac 场景。

### 半自动预抓取

8767 页面提供两步式辅助：

1. 选择目标，点击“生成并预览预抓取轨迹”；
2. 在 RViz 检查轨迹后，点击“确认移动到预抓取位”。

节点会按物体类型随机采样 8 个受约束候选，变化接近方位、仰角和预抓取距离，
并发布较短的可行 MoveIt 轨迹。执行完成后状态为 `HANDED_OVER`，后续微调、
闭合夹爪与抬升由人工完成。

命令行等价接口：

```bash
ros2 service call /smolvla/generate_pregrasp std_srvs/srv/Trigger "{}"
ros2 service call /smolvla/execute_pregrasp std_srvs/srv/Trigger "{}"
ros2 topic echo /smolvla/pregrasp_status --field data
```

注意：当前 MoveIt PlanningScene 只有机器人模型，尚未把全部桌面资产注册为碰撞
物。仿真节点通过抬高候选位置降低风险，但在真机上使用前必须补充场景碰撞物。

### `HANDED_OVER` 后的末端精调

在 8767 页面选择 2/5/10 mm 平移步长和 2/3/5° 旋转步长，然后使用按钮或
键盘：

```text
W/S  沿工具 X 轴接近/远离
A/D  沿工具 Y 轴左移/右移
R/F  沿基坐标 Z 轴上升/下降
Q/E  绕工具 X 轴逆转/顺转
```

另有打开/闭合夹爪、抬升 20 mm 和停止按钮。每次按键都是独立 MoveIt 微步，
必须等待状态重新显示 `HANDED_OVER` 再发下一步。当前实现不是连续速度 Servo，
因为本机未安装 `moveit_servo`。

### MoveIt 桌面和蓝盒碰撞物

一键 launch 在 `moveit:=true` 时启动 `planning_scene_sync`，向 MoveIt 注册：

```text
smolvla_table
smolvla_box_bottom
smolvla_box_left
smolvla_box_right
smolvla_box_front
smolvla_box_back
```

GUI 显示“已载入 6 个”即表示 `/apply_planning_scene` 已接受。完整验证命令：

```bash
ros2 service call /get_planning_scene \
  moveit_msgs/srv/GetPlanningScene \
  "{components: {components: 24}}"
```

手动重新载入：

```bash
ros2 service call /smolvla/reload_collision_scene std_srvs/srv/Trigger "{}"
```

动态物体由 `object_collision_sync` 自动加入 PlanningScene：

```text
smolvla_object_cup
smolvla_object_apple
smolvla_object_peach
smolvla_object_mouse
smolvla_object_beverage_bottle
```

节点订阅 `/isaac/object_transforms` 持续更新位置，并订阅 `/smolvla/status`
判断当前目标及抓稳状态。只有当前目标允许接触 `link06`、
`gripperStator`、`gripperMover`；抓稳后目标成为挂在 `gripperStator` 上的
`AttachedCollisionObject`，释放后恢复为世界碰撞体。GUI 的“动态物体碰撞”
应显示“已同步 5 个”或“已附着 目标”。

状态与手动重载命令：

```bash
ros2 topic echo /smolvla/object_collision_status
ros2 service call /smolvla/reload_object_collisions std_srvs/srv/Trigger "{}"
```

### SmallVLA episode 录制与 LeRobot 转换

`episode_recorder` 默认随一键 launch 启动，以 10 Hz 记录：

```text
observation.images.wrist   腕部 RGB
observation.images.global  全局 RGB
observation.state          7 维实测关节位置（含夹爪）
action                     7 维实际发送的关节目标
task                       英文语言指令
phase/grasp_success/task_success
各数据源 ROS 时间戳
```

默认原始数据目录：

```text
/home/lp531/smolvla_data/z1_tabletop_raw
```

推荐从 8767 GUI 的“SmallVLA 训练集录制”卡片操作。也可直接调用：

```bash
ros2 service call /smolvla/recording/start std_srvs/srv/Trigger "{}"
ros2 service call /smolvla/recording/save std_srvs/srv/Trigger "{}"
ros2 service call /smolvla/recording/discard std_srvs/srv/Trigger "{}"
ros2 topic echo /smolvla/recording_status
```

默认只有任务成功且不少于 10 帧才能保存。写入期间使用
`.recording_NNNNNN`，保存时原子重命名为 `episode_NNNNNN`，避免中断数据被
误当作训练样本。每条 episode 包含 `frames.jsonl`、双相机 JPEG、开始配置和
最终摘要。

正式训练前转换为本机 LeRobot 0.3.4 格式：

```bash
conda run -n unifolm-vla python \
  /home/lp531/isaac_assets/z1/scripts/export_z1_lerobot_dataset.py
```

默认输出 `/home/lp531/smolvla_data/z1_tabletop_lerobot`。转换器只选择成功
episode，生成双路 MP4、Parquet、任务表和统计量，并在结束时重新加载数据集
验证 episode、帧数与 FPS。输出目录已存在时会拒绝覆盖，应通过 `--output`
指定新目录。

## 11. 启动方式

### 11.1 推荐：一个命令启动 Isaac、MoveIt 和 RViz

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

默认值：

```text
start_isaac:=true
moveit:=true
rviz:=true
headless:=false
```

因此不需要再手动启动 Isaac。

### 11.2 无界面联调

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  headless:=true \
  moveit:=false \
  rviz:=false
```

### 11.3 Isaac 已经手动启动

ROS 侧：

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  start_isaac:=false
```

Isaac 侧必须清除系统 ROS Python 路径，让 Isaac 使用自带 Python 3.11 和
自带 Humble bridge：

```bash
env \
  -u PYTHONPATH \
  -u AMENT_PREFIX_PATH \
  -u COLCON_PREFIX_PATH \
  -u CMAKE_PREFIX_PATH \
  ROS_DISTRO=humble \
  RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  LD_LIBRARY_PATH=/home/lp531/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib \
  /home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py \
  --scene /home/lp531/isaac_assets/z1/generated/smolvla_red_ball_into_box.usd
```

### 11.4 只打开场景，不连接 ROS

```bash
/home/lp531/isaac-sim/isaac-sim.sh \
  /home/lp531/isaac_assets/z1/generated/smolvla_red_ball_into_box.usd
```

## 12. 验证步骤

### 12.1 场景烟测

```bash
/home/lp531/isaac-sim/python.sh \
  /home/lp531/isaac_assets/z1/scripts/smoke_test_smolvla_scene.py
```

期望：

```text
status: ok
frame_shape: [480, 640, 4]
```

并生成：

```text
generated/wrist_color_preview.png
```

### 12.2 ROS 图检查

```bash
ros2 node list
ros2 topic list
ros2 action list -t
ros2 topic echo /joint_states --once
ros2 topic echo /camera/camera/color/camera_info --once
```

### 12.3 机械臂 Action 独立测试

```bash
ros2 action send_goal \
  /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {
    joint_names: [joint1, joint2, joint3, joint4, joint5, joint6],
    points: [{
      positions: [0.2, 0.5, -0.7, 0.3, 0.2, -0.1],
      time_from_start: {sec: 3, nanosec: 0}
    }]
  }}"
```

已验证结果：

```text
目标: [0.2, 0.5, -0.7, 0.3, 0.2, -0.1]
实测: [0.2, 0.5007, -0.6953, 0.3036, 0.1999, -0.1]
最大误差: 0.0047 rad
状态: SUCCEEDED
```

### 12.4 RViz

在 MotionPlanning 面板中使用 `Plan & Execute`。只点击 `Plan` 只显示预览，
不会向 Isaac 发送轨迹。半透明机器人通常是规划起点/终点或轨迹预览，不是
Isaac 的实时机器人状态。

### 12.5 抓取判定器

```bash
ros2 topic echo /isaac/object_transforms --once
ros2 topic echo /smolvla/status --field data
ros2 topic echo /smolvla/task_phase
ros2 topic echo /smolvla/grasp_success
ros2 topic echo /smolvla/task_success
```

2026-07-21 使用独立 ROS 域注入完整合成 episode，实际观察到：

```text
WAITING_FOR_STATE
→ READY
→ GRASPING
→ TRANSPORT       grasp_success=true
→ PLACING
→ SUCCESS         task_success=true
```

同时已用真实 Isaac 场景验证五个物体 frame、实时物体位置/速度、夹爪角度、
目标切换以及重置服务。合成 episode 只用于覆盖判定状态机，不修改 USD 场景。

## 13. 常见问题

### `MoveGroupInterface::move() failed or timeout reached`

检查是否仍在运行旧进程。修改代码、重新生成 USD 或重新编译后，必须将原
launch `Ctrl+C` 完全退出并重新启动。

然后检查：

```bash
ros2 action list -t
ros2 topic hz /joint_states
ros2 topic echo /clock --once
```

### RViz 和 Isaac 最终位姿不一致

必须同时满足：

- 相同关节名与顺序；
- 轨迹桥 `use_sim_time=true`；
- `/clock` 正常；
- 关节驱动使用本文增益；
- MoveIt 没有在 Isaac 收敛前取消目标；
- 观察的是 RViz 实时 RobotModel，而不是半透明规划预览。

同一关节角下，ROS URDF 与 USD 的 `link00 -> gripperStator` 正运动学已经
验证一致，因此不要通过交换关节顺序或添加任意坐标旋转来“校正”该问题。

### 相机图像正常但 CameraInfo 不一致

Isaac 原始值在：

```text
camera_info_raw
```

训练和 ROS 算法应读取：

```text
camera_info
```

确认 `/camera_info_fix` 节点存在。

### Isaac 启动时报找不到 `rclpy`

不要让 Isaac Python 继承系统 ROS 的 `PYTHONPATH`。使用本指南 11.3 的环境
清理命令，或直接使用一键 launch。

### URDF 导入出现空 visual reference 警告

Isaac 5.1 importer 会为纯 TF 空 link 生成无效 visual reference。构建脚本
会在导入后清理十个此类引用。导入过程中的临时警告可以出现，但最终构建和
烟测必须成功。

## 14. 当前关键文件 SHA-256

用于判断以后是否仍是本次验证版本：

```text
1d50f3b81a3da3a49018908f9c7f8454ddbb8a6332c3bf5d3deb116b4ffd657d  scripts/build_smolvla_red_ball_scene.py
e33f48a4ff3abe543b5f4c7ab4e573db962116dd664e836bfb13b289fe919c55  scripts/prepare_scanned_assets.py
5909a8a92f0f9d6c5cfe6a70156f88c7649e84c0de3afb53d8f3c75e04568ab0  scripts/run_z1_ros2_scene.py
e06a4561c8e4a9018a497a020c5e57826cf62325e5240cc1633bc5035d0fc0c9  scripts/smoke_test_smolvla_scene.py
80d0d4104e10e9f01fa7c66ba3835fcd812db663112184f2793092c6db2913e0  generated/z1_with_d435.urdf
6a7a56bc6eaf5bdf647e28b514dffd1b52789e7d8aeed8542e11eee84915f435  generated/z1_with_d435.usd
0bfc7b8a7840c369a0c9764b6d3d11bb3025fc59b34879c21752ef5b3ff7b147  generated/smolvla_red_ball_into_box.usd
a0744790fb9511052d155ea3f9036691f81ea9513ce8f44a7afca713237cd608  z1_isaac_bridge/trajectory_bridge.py
12ac339f194b9add552208482946b1a31b472fb78de4fa52086515edae330c9a  z1_isaac_bridge/grasp_evaluator.py
46b105ed97773767d7bc0bb27b948a98c97c14a3c446765901f728d5e209381e  z1_isaac_bridge/launch/z1_isaac.launch.py
491482c69dbba62e403fb1e8a3b37472fcf5729072836001dd9a89988ab0fe9e  z1_moveit/launch/move_group.launch.py
```

重新构建 USD 后，生成文件哈希正常情况下会改变；脚本哈希未变而输出哈希改变
时，应先检查 Isaac Sim 版本、外部资产和 Xacro 是否一致。

## 15. 路径迁移

当前脚本和 launch 中存在 `/home/lp531` 绝对路径。迁移到其他用户名或机器时，
至少搜索并替换：

```bash
rg -n "/home/lp531" \
  /新的路径/isaac_assets/z1 \
  /新的路径/ros2_ws/src/z1_ros2
```

重点修改：

- `build_smolvla_red_ball_scene.py` 的根目录和 z1_description 路径；
- `prepare_scanned_assets.py` 的 `ROOT`；
- `run_z1_ros2_scene.py` 默认场景和 `ISAAC_SIM_ROOT`；
- `z1_isaac.launch.py` 默认场景、脚本和 Isaac Python；
- Xacro 注释中的标定来源路径不影响运行，但建议同步更新。

## 16. 推荐归档方式

在当前机器上进行可迁移备份时，应至少归档：

```text
ros2_ws/src/z1_ros2/
ros2_ws/z1_handeye_charuco_gripper_to_camera.yaml
isaac_assets/z1/scripts/
isaac_assets/z1/scanned_objects/
isaac_assets/z1/generated/
```

`ros2_ws/build`、`install`、`log` 可以重新生成，不是必须备份；但恢复后必须重新
执行 `colcon build --symlink-install`。
