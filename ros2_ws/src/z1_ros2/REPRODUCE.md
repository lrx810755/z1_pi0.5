# Z1 Isaac 项目复现指南

本仓库包含 ROS 2 代码、活动 Isaac 场景、五种训练物体和三相机录制流程。
USD、USDZ、DAE、STL 由 Git LFS 管理。Isaac Sim、训练数据、模型权重和
Unitree SDK 不进入 Git。

## 1. 克隆并取得资源

把仓库放在 ROS 工作区的 `src` 下：

```bash
sudo apt-get install -y git-lfs
git lfs install
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/lrx810755/z1_ros2.git
cd z1_ros2
git lfs pull
bash scripts/verify_reproduction.sh
```

若最后一条命令报告 LFS 指针或资源为空，先解决 Git LFS 下载问题，不要启动
Isaac。`isaac_assets/z1/scanned_objects/` 中的署名与来源文件必须保留。

## 2. Ubuntu 22.04 原生部署

先按照 ROS 2 官方方式安装 Humble Desktop、rosdep 和 colcon，并把 Isaac Sim
5.1 安装到仓库外。默认目录是 `~/isaac-sim`，也可用 `ISAAC_SIM_ROOT` 覆盖。

```bash
cd ~/ros2_ws/src/z1_ros2
./scripts/bootstrap_ubuntu_22_04.sh
source scripts/configure_environment.sh
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

脚本使用系统 ROS Python 包并设置 `PYTHONNOUSERSITE=1`，避免用户目录中的新版
setuptools/pytest 插件破坏 Humble 构建。默认场景为双腕相机场景，并发布腕部、
VLA 腕部和全局三个 RGB 流。

无图形界面的检查可使用：

```bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py \
  headless:=true rviz:=false moveit:=false
```

## 3. RTX 5070 Ti、Windows 11 与 WSL2

建议 Windows 运行 NVIDIA 驱动和 Windows 版 Isaac Sim 5.1，WSL2 Ubuntu
22.04 运行 ROS 2 Humble。Windows 与 WSL 各克隆同一提交：Windows 克隆供
Isaac 读取资源，WSL 克隆放在 Linux 文件系统的 `~/ros2_ws/src` 中用于构建，
不要在 `/mnt/c` 下执行 colcon。

管理员 PowerShell 中准备 WSL：

```powershell
wsl --install -d Ubuntu-22.04
wsl --update
```

Windows 11 支持时，可在 `%UserProfile%\.wslconfig` 中启用镜像网络：

```ini
[wsl2]
networkingMode=mirrored
firewall=true
```

然后执行 `wsl --shutdown`。确认 Windows 防火墙允许 Isaac Sim 的 UDP 通信，
并让两端使用相同的 `ROS_DOMAIN_ID`（启动脚本默认都是 0）。

在 WSL 中克隆、安装 ROS Humble 后执行：

```bash
cd ~/ros2_ws/src/z1_ros2
./scripts/bootstrap_ubuntu_22_04.sh
./scripts/start_ros_stack.sh
```

该脚本设置纯 UDP 的 `fastdds_wsl2.xml`，并以 `start_isaac:=false` 启动完整 ROS
配套节点；录制器和抓取评估器仍会运行。随后在 Windows PowerShell 的仓库中
启动 Isaac：

```powershell
.\windows\start_isaac_sim.ps1 `
  -IsaacSimRoot "C:\path\to\isaac-sim" `
  -RepoRoot "C:\path\to\z1_ros2"
```

可加 `-Headless`。若跨主机发现失败，先检查镜像网络、防火墙、两端
`ROS_DOMAIN_ID`，再检查 WSL 中的 `FASTRTPS_DEFAULT_PROFILES_FILE`。

## 4. 三相机和时间同步检查

启动后在 ROS 终端检查：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /vla_wrist_camera/color/image_raw
ros2 topic hz /global_camera/color/image_raw
python3 z1_isaac_bridge/test/camera_stamp_probe.py --duration 90
```

录制频率默认 10 Hz，图像同步容差 45 ms。当前方案关闭无人订阅的深度发布，
降低 GPU 到 CPU 像素回读和 DDS 压力；不要通过盲目放宽同步容差掩盖丢帧。
如任务确实需要深度，给 Isaac 脚本传入 `--depth-streams`。

## 5. 数据集与一键连续采集

数据集默认写到 `~/smolvla_data`，不进入 Git。可在启动前覆盖：

```bash
export Z1_DATASET_ROOT=/data/z1_datasets
```

完整仿真栈和任务 GUI 启动后，一键连续采集多轮成功 Episode：

```bash
ros2 run z1_isaac_bridge continuous_training \
  --target mouse --episodes 20 --curriculum easy
```

它会重试失败轮次，只计入已成功保存的数据。`--max-attempts 0` 表示不限重试，
Ctrl+C 会强停并丢弃未完成 Episode。

## 6. 可选 Python 环境

ROS/MoveIt、`rclpy`、消息包和 `cv_bridge` 由 apt/rosdep 管理；Isaac 模块由
Isaac Sim 自带，均不可替换为 PyPI 同名包。普通依赖见 `requirements/python.txt`。

YOLO 与训练环境应隔离：

```bash
python3 -m venv ~/venvs/z1-yolo
source ~/venvs/z1-yolo/bin/activate
python -m pip install -r requirements/yolo.txt
```

LeRobot 环境先按目标 NVIDIA 驱动选择匹配的 CUDA Torch，再安装：

```bash
python -m pip install -r requirements/training.txt
```

仓库固定 `lerobot==0.3.4`，但不固定 Torch CUDA wheel。

## 7. 真实机械臂与相机

真实 Z1 运行还需要仓库外的 Unitree Z1 SDK/控制器、机械臂网口配置、急停与
安全工作区。用 `Z1_UNITREE_ROOT`、`Z1_REAL_ARM_SDK_DIR` 和
`Z1_REAL_ARM_CONTROLLER_DIR` 指定位置。RealSense 实机还需对应 ROS 驱动，
并重新验证内参、手眼标定、桌面高度和碰撞模型。仿真通过不代表可直接安全驱动
真实机械臂。

## 8. 完整验收

```bash
cd ~/ros2_ws/src/z1_ros2
bash scripts/verify_reproduction.sh
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  z1_isaac_bridge/test z1_examples/test -q
cd ~/ros2_ws
PYTHONNOUSERSITE=1 colcon build --symlink-install
```

只读验证脚本检查资源完整性、五个源模型 SHA256、Git LFS 指针、Python/XML/
Shell 语法、活动路径、凭据模式以及缓存/备份误提交。它不会启动 Isaac 或机械臂。
