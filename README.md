# Unitree Z1 桌面抓取项目

本仓库归档了从 Z1、RealSense、ROS 2、MoveIt 和 Isaac Sim 工程搭建，到示范录制、多模型训练和闭环评估的项目。仿真主线测试苹果、桃子、杯子和饮料瓶的定位、抓取、抬升、搬运与蓝盒放置。实机视觉和语音控制作为历史分支保存。

[项目展示与可引用结果](docs/portfolio/README.md) · [复现说明](docs/reproduction/README.md) · [试错与历史索引](docs/history/README.md) · [外部数据与权重清单](manifests/external_artifacts.json)

| 目录 | 内容 |
| --- | --- |
| [ros2_ws/src/z1_ros2](ros2_ws/src/z1_ros2) | 活动 ROS 工程、机械臂和相机描述、Isaac USD 与场景脚本、控制/抓取/录制节点、物理评估与测试 |
| [robot](robot) · [SH](SH) | 示范采集、纠偏、质量筛查、四物体评估、语音控制与运行入口 |
| [pi05_lora_workspace](pi05_lora_workspace) | PI0.5 LoRA、ACT 和其他试验的源码、配置、冻结清单、转换和验证记录 |
| [smolvla_training](smolvla_training) | 历史训练报告、选定检查点配置与处理器；实际权重见外部资产清单 |
| [smolvla_eval](smolvla_eval) · [act_eval](act_eval) · [video_exports](video_exports) | 实验设置、逐次结果、报告、对照脚本、视频章节与导出脚本 |
| [z1_queue_logs](z1_queue_logs) | 采集/重录审计、替换映射、参数变化和调试记录 |
| [history](history) · [docs/history](docs/history) | 早期 ROS、Isaac、SmolVLA 工程快照、实机 YOLO 与标定、OpenVLA 本地探索和项目交接 |
| [manifests](manifests) · [archive_tools](archive_tools) | 原始路径与校验、依赖版本、外部资产位置、归档验证工具 |

最终 PI0.5 训练冻结了 **1,762 条轨迹**和三路 **1,703,757 张 JPEG**，完成 **100,000 次优化器更新**，通过最终模型重载检查。具体验证与闭环表现分别见 [训练完成记录](pi05_lora_workspace/pi05_current_round_20261111/completed.json) 和 [评估结果索引](docs/portfolio/README.md)。

原始示范、官方底座、LoRA 权重、全量视频和大型遥测保存在原始位置。仓库登记了需要恢复的目录、关键权重校验值以及已冻结图像的哈希。历史源码与报告保留原有绝对路径。跨机复现先按 [恢复布局说明](docs/reproduction/README.md) 配置目录。早期文档中的状态以文档日期为准。
