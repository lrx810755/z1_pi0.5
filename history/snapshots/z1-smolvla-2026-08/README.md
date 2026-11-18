# Z1 SmolVLA

这是 Unitree Z1 在 Isaac Sim、ROS 2、MoveIt、LeRobot 和 SmolVLA 上的桌面抓取训练工程快照。

本目录由现有工作目录复制整理而成，未修改原始工程文件。为避免破坏 ROS 2、Isaac Sim 和 Python 包的内部结构，仓库保留三套规范布局：

- `ros2_ws/src/z1_ros2/`：Z1 描述、MoveIt、控制桥、抓取和录制节点；
- `isaac_assets/z1/`：Isaac 场景、物理资产、运行脚本和工作记录；
- `lerobot/`：固定版本的 LeRobot/SmolVLA 训练代码。

顶层的 `robot_description`、`simulation`、`ros2_control`、`grasping`、`teleoperation`、`dataset`、`training` 和 `tests` 是模块入口。其中大部分是指向上述规范布局的相对软链接，不会重复保存源文件。

## 模块

| 模块 | 入口 | 作用 |
|---|---|---|
| 机器人描述 | `robot_description/` | Z1 URDF/Xacro、网格、夹爪和相机链 |
| 仿真环境 | `simulation/` | Isaac USD、扫描资产、场景构建和运行 |
| ROS 2 控制 | `ros2_control/` | 轨迹桥、复位、PlanningScene 和启动文件 |
| 抓取系统 | `grasping/` | 闭环对中、物理接触、试抬、搬运和放置 |
| 自动示教 | `teleoperation/` | 8767 前端后端、任务编排和软件强停 |
| 数据管线 | `dataset/` | Episode 录制和 LeRobot 数据转换 |
| 模型训练 | `training/` | LeRobot 与 SmolVLA 模型、训练入口 |
| 测试 | `tests/` | 抓取、录制和自动示教测试 |
| 文档 | `docs/` | 工作记录、复现指南和训练管线说明 |

更完整的依赖关系见 `ARCHITECTURE.md`。

## 数据

1.7 GB 的历史数据没有复制进代码仓库，原始位置仍是：

```text
/home/lp531/smolvla_data
```

数据现状和迁移方式见 `data/README.md`。正式训练前必须人工审核 Episode，不能把未完成的 `.recording_*` 或历史无效回合直接作为训练集。核心迁移包只包含复现所需代码与资产，数据需要单独复制。

## 可移植路径

活动源代码不再依赖固定用户名。运行前执行：

```bash
source scripts/setup_env.sh
```

脚本会自动设置仓库、ROS 工作空间、Isaac 资产、数据和 Isaac Sim 路径。需要自定义时可覆盖 `.env.example` 中列出的环境变量。完整步骤见 `REPRODUCE.md`。

## 常用入口

完整系统：

```bash
source scripts/setup_env.sh
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${Z1_ROS_WS}/install/setup.bash"
ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

数据转换：

```bash
python "${Z1_ASSET_ROOT}/scripts/export_z1_lerobot_dataset.py" \
  --help
```

SmolVLA 训练入口：

```bash
lerobot-train --help
```

当前尚未固化项目专用的一键正式训练配置。
