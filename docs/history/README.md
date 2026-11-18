# 工程演进与试错证据

| 阶段 | 资料与记录 |
| --- | --- |
| 6–8 月，机械臂与仿真搭建 | [早期红球实验](../../history/isaac_assets/z1/scripts/z1_grasp_experiment_notes.md)、[从 URDF 到训练数据](../../history/snapshots/z1-smolvla-2026-08/docs/Z1_URDF_TO_SMOLVLA_TRAINING_PIPELINE.md)、[8 月复现快照](../../history/snapshots/z1-smolvla-2026-08) |
| 实机视觉探索 | [YOLO + D435 分支](../../history/hardware/z1_yolo)、[标定资料](../../history/hardware/calibration)、[接近日志](../../history/hardware/monitoring/grasp_approach_work_log.md) |
| 9 月，物理验收与采集 | [双指夹持验证](../../ros2_ws/src/z1_ros2/z1_isaac_bridge/PHYSICAL_GRASP_VALIDATION.md)、[相机传输和时间戳](../../ros2_ws/src/z1_ros2/docs/camera_sync_transport_fix_2026-09-12.md)、[早期 SmolVLA 失败分析](../../smolvla_training/SmolVLA训练与验证失败分析_20260928.md) |
| 10 月，闭爪及模型/执行对照 | [PI0.5 160 条微调记录](../../smolvla_training/PI0.5_160条视频_LoRA微调与评估总结_20261008.md)、[状态与时序核查](../../smolvla_eval/pi05_input_sync_fix_20261005)、[ACT 接近与执行窗口](../../smolvla_eval/act_apple_8vs15_20261018_173608)、[杯子物理对照](../../z1_queue_logs/cup_cylinder_comparison_20261026) |
| 11 月，质量筛查与冻结训练 | [筛查工具和裁决](../../robot/dataset_trajectory_review)、[冻结清单](../../pi05_lora_workspace/pi05_current_round_20261111/split_manifest.json)、[训练来源与标签审计](../../pi05_lora_workspace/pi05_current_round_20261111/final_label_timeline_audit.json)、[迁移交接](帐号迁移交接_20261115.md) |
| 11 月 17–18 日，最终闭环 | [原始 15 步组](../../smolvla_eval/pi05_final_four_objects_50each_20261117_145347)、[碰撞过滤 15 步组](../../smolvla_eval/pi05_final_execute15_finger_table_filtered_30each_20261118)、[停止的 20 步组](../../smolvla_eval/pi05_final_filtered_execute20_30_30each_20261118/execute_20)、[完成的 30 步组](../../smolvla_eval/pi05_final_filtered_execute30_15each_20261118) |

[42 项问题与解决方法](../../pi05_lora_workspace/docs/项目全过程复盘_20261113/01_问题与解决方法.md)和[12 阶段决策回顾](../../pi05_lora_workspace/docs/项目全过程复盘_20261113/02_思考与决策过程.md)整理了 11 月 13 日前的记录。旧文档以当时状态为准。项目证据的当前归档路径和 SHA256 见[复盘证据索引](../../manifests/retrospective_sources.json)。原始会话缓存保留在个人环境，仓库使用已整理的工程决策记录。

试验记录应保留假设、变更、控制变量、终止条件和结果。腕部光学遮挡和真实几何接触、训练预测长度与运行执行长度、完整试次和人为停止的试次均有各自记录，可沿索引回到原报告。
