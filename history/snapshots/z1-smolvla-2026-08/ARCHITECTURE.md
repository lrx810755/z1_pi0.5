# 工程架构

```text
Z1 Xacro / URDF / 标定
            |
            v
Isaac Sim 机器人与任务场景
            |
            v
ROS 2 / MoveIt / 轨迹桥
            |
            v
闭环抓取与 PhysX 物理验证
            |
            v
自动示教与 Episode 录制
            |
            v
LeRobot 数据集转换
            |
            v
SmolVLA 微调与策略评估
```

## 规范源位置

同一功能可能同时出现在顶层模块入口和规范源位置。编辑或比较文件时，应以规范源位置为准：

| 功能 | 规范源位置 |
|---|---|
| Z1 描述 | `ros2_ws/src/z1_ros2/z1_description/` |
| MoveIt | `ros2_ws/src/z1_ros2/z1_moveit/` |
| Isaac/ROS 桥 | `ros2_ws/src/z1_ros2/z1_isaac_bridge/` |
| Isaac 场景 | `isaac_assets/z1/` |
| 自动示教后端 | `isaac_assets/z1/scripts/z1_teleop_backend/` |
| 数据转换器 | `isaac_assets/z1/scripts/export_z1_lerobot_dataset.py` |
| SmolVLA | `lerobot/src/lerobot/policies/smolvla/` |

顶层相对软链接只用于导航，不是第二份源码。

## 外部运行依赖

- Ubuntu 22.04；
- ROS 2 Humble 与 MoveIt；
- Isaac Sim 5.1.0；
- NVIDIA 驱动和 CUDA 兼容的 PyTorch；
- Python 3.10 LeRobot/SmolVLA 环境；
- SmolVLA 基础 checkpoint。

这些大型运行时不应提交到本仓库。

