# Episode 复位与单次定位抓取修改（2026-09-07）

## 行为

- 复位控制权交接继续保留：复位期间停止发送旧保持命令，接触和关节反馈更新、夹指离桌、控制器停止后才清故障。
- 清故障失败分别报告关节反馈年龄、接触反馈年龄、桌面接触、缺失关节和占用控制器；失败不会解除复位独占状态。
- 手动复位成功后清除旧自动示教错误，界面回到 IDLE；复位失败或示教仍运行时不覆盖错误/运行状态。
- 苹果、桃子使用现有 `desired_center_x/y/z_m` 与末端到 `gripperStator` 的实际固定 TF 反解最终末端位置，不需要人工参考点。生成并执行一次最终定位轨迹后，恢复动态物理，只 probe 验收一次；任一轴超过 4 mm、越出夹持包络、缺失接触反馈、夹指已触物或碰桌均停止。验收通过后进入既有闭爪、有限预载、真实双指接触及物理试抬验证。关闭旧外侧对中、动态插入、闭爪后横向搜索和预载期间解卡重定位分支。
- 杯子与瓶子共用 `_guided_single_pass_cylinder_grasp`。杯子沿用瓶子的工具 roll 和默认 pitch（URDF -5°），候选 pitch 仅在 -5° 至 -7° 之间变化。采用 61 mm 杯身、48 mm 上部抓取高度，双指有效接触带为距杯底 38–57 mm。根据瓶子的工具相对抓取带高度偏移反解杯子姿态；规划时几何中心与抓取高度分别处理，避免把 30.5 mm 中心当作夹持高度。该高度是几何推算初值，并非实测标定。
- 杯子始终开启定位验收，不受瓶子的调试开关影响，也不会启用瓶子的 Servo Y 校正。动态验收再次检查误差和物理状态，保持 MoveIt 碰撞检查、夹指碰桌停止、真实双指受力和离桌验证。

## 修改文件

- `z1_isaac_bridge/z1_isaac_bridge/trajectory_bridge.py`
- `z1_isaac_bridge/z1_isaac_bridge/grasp_gui_bridge.py`
- `z1_isaac_bridge/z1_isaac_bridge/grasp_assistant.py`
- `z1_isaac_bridge/z1_isaac_bridge/grasp_config.py`
- `z1_isaac_bridge/config/grasp_physics.json`
- `z1_isaac_bridge/test/test_single_position_grasp.py`
- `z1_isaac_bridge/test/test_motion_protection.py`
- `z1_isaac_bridge/test/test_grasp_physics.py`

## 验证

140 项相关测试通过，覆盖坐标变换、单次验收成功/失败分支、超差时不得闭爪、杯子高度补偿与杯底/杯沿接触拒绝、故障保持和复位界面状态。

构建命令：

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 colcon build --packages-select z1_isaac_bridge
```

后端已重启加载更新。此次重启发现旧 Isaac 子进程未随旧 ROS 启动器退出，已清理；第二次采用先终止 Isaac、再退出启动器的顺序，避免旧仿真时钟污染新 TF。

复位服务实测返回成功，界面为 `episode_reset=COMPLETE`、`auto_demonstration=IDLE`。

未执行三类目标的完整闭爪、试抬和放置物理回归，不能据此声称实际抓取成功；单次定位若实际执行超差，会按设计停止并输出偏差。

### 在线规划检查

| 目标 | 结果 |
| --- | --- |
| 苹果 | 最终定位轨迹生成成功，末端 `(0.394, -0.178, 0.206) m`，最大关节变化 `2.082 rad` |
| 杯子 | 切换目标并复位后最终定位轨迹生成成功，末端 `(0.448, -0.077, 0.101) m`，最大关节变化 `2.442 rad` |
| 桃子 | 默认布局及 easy 随机布局（seed `951111561`）均被苹果占用接近扫掠通道，8 个候选被拒绝；未放宽碰撞检查 |

首次连续切换目标而未逐目标复位时，杯子报距夹爪 1.299 m，冻结门限拒绝；恢复场景后同一门限下规划通过。该检查不代表杯子已完成物理抓取。最后再次复位成功，保留场景待下一轮操作。
