# 从归档复现 Z1 抓取

本说明提供三条路线：重建场景和采集接口，重跑已有模型评估，从冻结示范重新训练。历史目录保存原版配置和证据，重跑放到新的输出目录。旧版本使用不同的相机、物理和成功判据，按各自报告复现。

## 1. 检查归档和环境

```bash
git lfs install
git lfs pull                     # 从已配置的远端克隆后执行
python3 archive_tools/verify_archive.py
```

原环境是 Ubuntu 22.04、ROS 2 Humble、Isaac Sim 5.1 和 RTX 4090。ROS/推理客户端使用 Python 3.10，独立模型工作进程使用 Python 3.12，Isaac 使用随安装提供的 Python。具体包版本与代码版本见 [运行环境](../../manifests/runtime_versions.json) 和 [源码版本](../../manifests/software_sources.json)。ROS、Isaac 和 CUDA 重新安装，构建目录由目标机器生成。

活动源码、旧的运行命令及 SHA 配置中保留了 `/home/lp531`。要原样复现，最直接的方法是在一台新机器或隔离环境中建立该项目根目录。把仓库内容放在该路径，使 `ros2_ws/src/z1_ros2`、`pi05_lora_workspace`、`robot`、`SH` 保持相对位置。Git 克隆可先放在任意位置，先运行只读验证，再在空的复现工作区恢复这些路径。若改用其他根目录，应统一迁移运行配置及其引用，并重新生成路径相关的校验和验证证据。保留仓库里的原版清单用于追溯。

外部依赖见 [资产清单](../../manifests/external_artifacts.json)。活动 USD 和网格已归档，旧版数据和模型对象按清单恢复。其中最终模型部署需要：

- `pi05_lora_workspace/weights/pi05_base`。用 [下载脚本](../../pi05_lora_workspace/download_base.py) 取得指定 revision 和 SHA 的底座。
- `smolvla_training/pi05_current_round_20261111/checkpoints/800000/pretrained_model/adapter_model.safetensors`。从本地备份或单独的数据存储恢复，核对 SHA256。其配置、tokenizer 和处理器已归档。
- 建议完整恢复冻结的 1,762 条示范，包括元数据和三路 JPEG。评估会使用示范摘要计算坐标覆盖和种子排除规则，重载验证还会读取八个数据池的真实 RGB 样本。原始 episode 文件由 SHA 校验，需与清单一致。最终模型和训练都使用同一冻结版本。

准备 LeRobot 与 OpenPI 时使用清单中的提交。保留官方基础模型和本项目 LoRA 的版本区别。不要直接把 ROS 的 Python 包安装到 Isaac 或模型的解释器里。

## 2. 启动场景和重新录制

场景文件在 [generated_dual_camera](../../ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera)，入口为 `smolvla_dual_wrist_camera.usd`。相机和物理参数在 [ROS 配置](../../ros2_ws/src/z1_ros2/z1_isaac_bridge/config)，最终四物体评估使用 [独立场景配置](../../robot/pi05_four_objects_eval_20261117/grasp_physics.json)。用户名、路径及 DDS 域按原环境一致设置。构建方式可参考 [活动 ROS 工程说明](../../ros2_ws/src/z1_ros2/REPRODUCE.md)。

在已经安装 ROS 2 Humble 的系统构建工作区：

```bash
cd /home/lp531/ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y
PYTHONNOUSERSITE=1 colcon build --symlink-install
source install/setup.bash
```

安装 Isaac Sim 之后，使 `ISAAC_SIM_ROOT` 指向安装目录，并配置测评脚本引用的 MoveIt Servo 前缀。先启动仿真，确认三路彩色图像、`/joint_states`、时钟、对象状态及复位服务。录制测试先走一条完整的抓取、抬升、放置轨迹，核对三路帧与关节/命令时间对齐，且 `episode_summary.json` 有真实物理验收。

录制代码见 [episode_recorder.py](../../ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/episode_recorder.py) 和 [recording_sync.py](../../ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/recording_sync.py)。历史各物体的采集入口在 [SH](../../SH)，筛查在 [dataset_trajectory_review](../../robot/dataset_trajectory_review)。复现采集时给新数据一个单独的输出路径。

## 3. 运行现有模型

环境、来源元数据、底座和 LoRA 就绪后，先检查，再刷新目标机器的模型加载证据。旧证据含文件身份，复制后的 inode/时间戳可能变化：

```bash
bash /home/lp531/SH/pi05_verify_final.sh
bash /home/lp531/SH/pi05_eval.sh --check
```

用一个终端运行 `bash /home/lp531/SH/pi05_eval_scene.sh --headless --output /绝对路径/新评估目录`，等待相机、复位、碰撞同步就绪。另一个终端运行：

```bash
bash /home/lp531/SH/pi05_eval.sh --run --objects all \
  --episodes-per-object 1 --execute-steps 15 --output /绝对路径/新评估目录
```

模型预测 30 步。执行 15、20 或 30 步后重新观测，动作 10 Hz、仿真控制插值 60 Hz。成功判据包括物理抓取、搬运、释放与稳定放置，不同阶段的结果与三视角录像可单独检查。

要重跑过滤夹爪/桌面碰撞的配对试验，从归档生成一份新运行目录。原始试次表自动相随：

```bash
python3 archive_tools/prepare_replay.py filtered-30 --output /绝对路径/pi05_filtered30_replay
env -u PYTHONPATH -u PYTHONHOME /home/lp531/pi05_lora_workspace/.venv/bin/python \
  /绝对路径/pi05_filtered30_replay/job/run_job.py
```

可选 `filtered-15`（原计划 120 次并已完成）、`filtered-20`（原计划 120 次，历史运行停止于 89 次）、`filtered-30`（60 次并已完成）。脚本启动新场景之前需要释放原场景的 ROS 进程和 8868 端口。过滤组启动时验证两种支撑关系和实际碰撞对。原始 15 步深度字段包含复位过程，用 [按首次推理重算的逐种子结果](../../smolvla_eval/pi05_final_execute15_finger_table_filtered_30each_20261118/paired_comparison_from_first_inference.json) 作为其深度对照。

## 4. 重建冻结训练

依据 [外部数据清单](../../manifests/external_artifacts.json) 和 [split_manifest.json](../../pi05_lora_workspace/pi05_current_round_20261111/split_manifest.json) 恢复 18 个物理来源目录。数据版本等价性由 episode 文件哈希与 `image_pins.npz` 核对。复制文件会改变部分文件时间戳，数据读取器对变化的文件核对内容 SHA256。

同一冻结数据已经附有 `lowdim.npz`、`image_pins.npz`、`sampling_arrays.npz`、`train_stats.json` 和转换证明。可以先在隔离复现环境，用它们检查 CPU 数据和处理器。如果要重建派生数据，先归档原派生物与训练输出，在隔离副本运行：

```bash
bash /home/lp531/SH/pi05_prepare_dataset.sh
bash /home/lp531/SH/pi05_convert_dataset.sh
bash /home/lp531/SH/pi05_train.sh
```

从头重跑 `prepare` 需要恢复[候选来源表](../../manifests/external_artifacts.json)里的原始目录与目录级条数，包括之后被替换的候选条目。旧杯子的 200 条筛选集用[符号链接清单](../../manifests/cup_selection_symlinks.json)重建，不能把它当作 18 个物理最终来源中的新数据。队列证据、替换映射与旧杯子动作标签数组已归档。仅重跑 `convert` 可直接核对已有的 `sources_frozen.json`。已有训练输出目录含检查点元数据，正式训练入口会保护该目录，重训前把新环境复制来的旧输出整体移至独立历史位置，并保留原归档。训练入口执行 GPU 冒烟、验证后从指定底座开始正式 LoRA。原实验是 10 Hz，预测 30 步，单 GPU BF16，microbatch 2×累积 8，有效批 16，100,000 次优化器更新。全部冻结数据用于训练。运行参数见 [resolved_train_config.json](../../pi05_lora_workspace/pi05_current_round_20261111/resolved_train_config.json)。

归档验证检查文件完整性、冻结清单、现有报告和复现入口。跨机 ROS/Isaac 启动、GPU 模型运行与全量重新训练分别在目标机器执行验收。
