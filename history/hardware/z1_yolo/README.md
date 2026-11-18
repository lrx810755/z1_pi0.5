# Z1 实机 YOLO 抓取工作区

本仓库打包了 Unitree Z1 机械臂 + RealSense D435 实机 YOLO 识别抓取所需的 ROS 2
源码包，裁剪掉了 Isaac Sim 桥、手眼标定工具、仿真场景和试验记录，只保留真机运行链。

## 仓库结构

```text
src/z1_ros2/
  z1_bringup/            # 实机 bringup：控制器、launch
  z1_description/        # Z1 URDF 与网格（含 D435 相机模型）
  z1_examples/           # YOLO/像素检测、抓取 GUI、approach、launch
  z1_hardware_interface/ # 真机硬件接口（构建时自动拉取 Unitree SDK）
  z1_moveit/             # MoveIt 配置与 launch
```

YOLO 模型 `yolo11n.pt` 位于 `z1_examples/models/`，构建时会随包安装到
`share/z1_examples/models/`，无需再手动下载或指定绝对路径。

## 从源码构建

```bash
git clone <repo-url> z1_yolo
cd z1_yolo
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python -m pip install --no-deps -r src/z1_ros2/z1_examples/requirements-yolo.txt
colcon build --symlink-install
source install/setup.bash
```

`--no-deps` 是为了避免 Ultralytics 默认拉入的 `opencv-python` 覆盖 GUI 使用的
`opencv-python-headless`，产生 Qt/xcb 插件冲突。全新 Python 环境请先安装与 CUDA 匹配的
PyTorch，再按上面命令安装其余依赖（清单中已包含 headless OpenCV）。

## 运行前必做：提供手眼标定矩阵

抓取流程启动时会校验 gripper→camera 标定矩阵文件。目标设备上请准备自己的
`gripper_to_camera.yaml`（与源机文件格式一致），然后任选一种方式指定：

```bash
export Z1_HANDEYE_CALIBRATION=/absolute/path/to/gripper_to_camera.yaml
```

或运行时加参数：

```bash
ros2 run z1_examples red_target_gui.py \
  --calibration /absolute/path/to/gripper_to_camera.yaml
```

若机械臂/相机相对安装关系与源机一致，也可以直接使用源机标定文件；若重新安装过相机，
请先在新设备完成标定再运行。

## 运行

一键启动（MoveIt + RealSense + 抓取 GUI）：

```bash
source install/setup.bash
export Z1_HANDEYE_CALIBRATION=/absolute/path/to/gripper_to_camera.yaml
ros2 launch z1_examples red_ball_real_grasp.launch.py
```

YOLO 识别接近抓取流程（approach 节点）：

```bash
ros2 launch z1_examples red_target_real_handeye.launch.py \
  detector:=yolo yolo_class:="sports ball" execute:=false
```

仅启动识别 GUI：

```bash
ros2 run z1_examples red_target_gui.py \
  --detector yolo --yolo-model yolo11n.pt --yolo-class "sports ball" --yolo-conf 0.35
```

## 依赖与硬件

- Ubuntu 22.04 + ROS 2 Humble
- RealSense ROS 驱动（`realsense2_camera`，经 rosdep/apt 安装）
- Unitree Z1 机械臂，本机网口按官方说明配置，默认机械臂 IP `192.168.123.110`
  （见 `z1_hardware_interface/config/config.xml`）
- Python 依赖见 `z1_examples/requirements-yolo.txt`

`z1_hardware_interface` 在 CMake 配置阶段会自动从 Unitree GitHub 拉取
`z1_sdk` / `z1_controller`，首次构建需要网络。
