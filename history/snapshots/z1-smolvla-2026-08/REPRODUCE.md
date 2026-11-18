# 从零复现

## 0. 解压核心仓库

把唯一的核心压缩包复制到目标 PC 后执行：

```bash
tar -I zstd -xf z1-smolvla-core-YYYY-MM-DD.tar.zst
cd z1-smolvla
```

压缩包已经包含顶层 `z1-smolvla/` 目录，不依赖 GitHub。训练数据、基础 checkpoint 和外部软件需要按后续章节单独恢复。

## 1. 目标环境

当前工程基线：

- Ubuntu 22.04；
- ROS 2 Humble；
- MoveIt 2；
- Isaac Sim 5.1.0；
- NVIDIA RTX 4090 级别 GPU；
- Python 3.10 LeRobot/SmolVLA 环境。

Isaac Sim、ROS 2、NVIDIA 驱动、基础 SmolVLA checkpoint 和正式训练数据不包含在代码仓库中。

## 2. 配置路径

仓库可以放在任意位置：

```bash
source scripts/setup_env.sh
./scripts/verify_reproduction.sh
```

默认路径均从当前仓库自动派生：

```text
Z1_SMOLVLA_ROOT=<当前仓库>
Z1_ASSET_ROOT=<当前仓库>/isaac_assets/z1
Z1_ROS_WS=<当前仓库>/ros2_ws
SMOLVLA_DATA_ROOT=<当前仓库>/data/runtime
ISAAC_SIM_ROOT=$HOME/isaac-sim
```

如 Isaac Sim 或数据位于其他位置，在 source 前覆盖对应变量：

```bash
export ISAAC_SIM_ROOT=/opt/isaac-sim
export SMOLVLA_DATA_ROOT=/data/z1-smolvla
source scripts/setup_env.sh
```

## 3. 构建 ROS 2 工作空间

安装 ROS 2 Humble、MoveIt、rosdep 和 colcon 后运行：

```bash
./scripts/bootstrap_ros.sh
source "${Z1_ROS_WS}/install/setup.bash"
```

脚本只编译仿真所需的 `z1_description`、`z1_examples`、`z1_moveit` 和 `z1_isaac_bridge`。

构建脚本会设置 `PYTHONNOUSERSITE=1`，避免目标用户 `~/.local` 中的 Python 包覆盖 ROS Humble 自带的 `setuptools/packaging`。

`z1_hardware_interface` 会在线获取 Unitree SDK，不属于 Isaac 仿真复现的必要步骤。需要真机时应固定 SDK/controller 版本后再构建。

## 4. 创建训练环境

```bash
conda env create -f environment.yml
conda activate z1-smolvla
```

先根据目标 PC 的驱动安装兼容的 CUDA PyTorch，再从仓库根目录执行：

```bash
python -m pip install -r requirements-training.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
lerobot-train --help
```

`torch.cuda.is_available()` 必须为 `True` 后再开始训练。

## 5. 重建 URDF 和 USD 派生资产

首次迁移、路径变化、Xacro/网格/相机修改后执行：

```bash
source scripts/setup_env.sh
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${Z1_ROS_WS}/install/setup.bash"
./scripts/rebuild_generated_assets.sh
```

脚本会展开 Xacro、生成 resolved URDF、规范化扫描模型、重新导入机器人 USD、生成双腕部相机场景并执行相机 smoke test。

脚本默认要求 `nvidia-smi -L` 能识别 GPU。如果只需尽力重建 USD、并明确接受没有 GPU smoke test 的风险，可临时设置 `Z1_ALLOW_NO_GPU=1`。若 Isaac 报 change watch 的 `No space left on device`，这通常是 inotify 实例上限而非磁盘空间；检查 `fs.inotify.max_user_instances`，并由目标 PC 管理员按需提高系统上限。

不要手工搜索替换 `generated/` 和 `generated_dual_camera/` 中的旧路径。

## 6. 恢复数据

把独立数据归档解压到 `SMOLVLA_DATA_ROOT`。完整归档包含失败回合、隔离数据和 `.recording_*`，只能作为原始资料保全。

正式训练前建立人工审核后的只读数据版本，再转换：

```bash
python "${Z1_ASSET_ROOT}/scripts/export_z1_lerobot_dataset.py" \
  --input "${SMOLVLA_DATA_ROOT}/reviewed" \
  --output "${SMOLVLA_DATA_ROOT}/lerobot/z1_tabletop_v1" \
  --repo-id lp531/z1_smolvla_tabletop
```

## 7. 启动系统

```bash
source scripts/setup_env.sh
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${Z1_ROS_WS}/install/setup.bash"
ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

也可以临时覆盖路径：

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  isaac_sim_root:="${ISAAC_SIM_ROOT}" \
  dataset_root:="${SMOLVLA_DATA_ROOT}/z1_tabletop_raw_dual_wrist"
```

## 8. 复现验收

至少确认：

1. Isaac 场景和两个腕部相机正常；
2. `/clock`、`/joint_states` 和三路 RGB 持续更新；
3. MoveIt/RViz 与 Isaac 关节一致；
4. ROS 2 测试通过；
5. Episode 可以录制、失败丢弃、成功原子保存；
6. LeRobot 数据集可以重新加载；
7. CUDA 可用；
8. 未见随机 seed 上完成策略评估后，才将 checkpoint 标记为可用。
