# 本轮 PI0.5 最终模型四物体评估

入口已准备：苹果 `apple`、桃子 `peach`、杯子 `cup`、饮料瓶 `bottle`/`beverage_bottle`，`all` 表示依次评估四类。

本次只准备和检查接口，未启动 Isaac、GPU 模型或闭环评估。训练仍在进行时，检查会显示未就绪；不会回退到 95000 次模型或等待后台自动启动。

## 手动使用

只读检查（也是不带参数的默认行为）：

```bash
/home/lp531/SH/pi05_eval.sh --check
```

预览命令，不启动任何场景或模型；最终模型尚未生成时也可预览：

```bash
/home/lp531/SH/pi05_eval.sh --dry-run --objects all --episodes-per-object 1
```

训练结束、GPU 可用后，先核验最终模型。此步骤不启动仿真：

```bash
/home/lp531/SH/pi05_verify_final.sh
```

这个入口先运行本轮 `verify.py --reload-final`，对全部八个物体/基础纠偏池检查正式模型；再用实际部署的独立推理进程检查八组原尺寸 RGB 输入。必须两项都通过，才允许 `--run`。

随后在第一个终端启动本轮专用场景；保留终端运行，等待场景及相机就绪：

```bash
/home/lp531/SH/pi05_eval_scene.sh
# 无窗口模式：/home/lp531/SH/pi05_eval_scene.sh --headless
```

在第二个终端执行评估，例如四类各 20 次（默认值）：

```bash
/home/lp531/SH/pi05_eval.sh --run --objects all --episodes-per-object 20
```

单独评估，或先每类检查一次：

```bash
/home/lp531/SH/pi05_eval.sh --run --objects apple --episodes-per-object 1
/home/lp531/SH/pi05_eval.sh --run --objects peach --episodes-per-object 1
/home/lp531/SH/pi05_eval.sh --run --objects cup --episodes-per-object 1
/home/lp531/SH/pi05_eval.sh --run --objects bottle --episodes-per-object 1
```

`--timeout 60` 是每条轨迹的仿真秒数。可用 `--output /绝对路径` 选择结果目录。固定布局使用 `--eval-seeds SEED...`，数量须与每物体次数一致且互不重复；四类复用同一 seed 列表，种子必须未用于任何训练来源，并通过实际布局范围及物理稳定性检查。

`--execute-steps 10`、`--execute-steps 20`、`--execute-steps 25` 或 `--execute-steps 30` 可选择执行窗口，模型仍预测训练时的30步，默认执行15步。2026-11-17新增的25/20步两组对比各为四物体各25次，30步及10步组另按四类各10次运行；这是执行窗口对比，不改模型检查点的预测长度。

停止评估用评估终端 Ctrl+C；运行时会关闭策略控制。场景终端独立保留，完成后也可 Ctrl+C 关闭。不启动自动采集、专家演示队列或语音控制。

## 模型、输入和执行契约

- 仅允许 `/home/lp531/smolvla_training/pi05_current_round_20261111/checkpoints/800000/pretrained_model`，对应 **100000 次参数更新**。要求 `completed.json` 和 `latest_checkpoint.json` 一致，不允许手工指定旧模型替代。
- 模型重载证据必须对应本轮冻结准备工件；部署桥接核验记录完整检查点 SHA256、底座 SHA256 和文件身份。工件变动后必须重新验证。
- 三路 `wrist/global/global_right`，原尺寸分别为 640×360、640×480、640×480；BGR 转 RGB 后由模型保存的处理器补边到 640 正方形并缩放到 224。不会重复裁剪。
- 状态为实际连续七维关节角；动作前六维为绝对关节目标，夹爪为闭合意图。使用保存的共享 q01/q99 归一化，不重算统计。
- 预测 30 步、每次执行前 15 步、动作 10Hz、命令插值 60Hz、flow 10 步。
- 沿用现有二值夹爪滞回：≥0.6 闭、≤0.4 开、中间保持前次意图；开 -0.9rad、闭 0rad，默认不增加闭爪最短保持时间。
- 场景使用采集配置中的苹果 1.1 倍、桃子 1.2 倍；独立 ROS 域 **168**、GUI 端口 **8868**。两终端均使用上述 SH；不要混入其他实验场景。
- 随机布局排除冻结清单全部 1762 条记录的训练种子；覆盖单元从实际坐标计算，不按采集目录名推断。不默认限制在示范点 3cm 内。
- 保留自然物理，不冻结目标至闭爪、不为模型调用专家修正。保留原执行器的关节限幅、跟踪误差、相机同步与陈旧输入检查、手指撞桌停止和瓶子倾倒停止。同步容差为 45ms；推理期间物理继续运行，保留现有观测老化局限。

## 结果

默认保存到 `/home/lp531/smolvla_eval/pi05_four_objects_final_20261117/pi05_sim_eval_时间/`：

- `report.json`：按物体统计成功数、总数、成功率和失败类型，逐条更新。
- `episodes/.../result.json`、`trace.jsonl`、`prediction_chunks.jsonl`、`input_sync.jsonl`：结果、命令、夹爪预测和输入时间证据。
- `episodes/.../videos/`：三视角 MP4。
- `stage_report.json`：本次套件结束后汇总闭爪、双侧接触、抬升、稳定抓取、搬运至盒、释放和稳定放置的观测证据；缺失证据记为 unknown。
- `evaluation_provenance.json`：模型、冻结来源范围与代码哈希。

最终成功沿用物理评估器的稳定放置 `task_success`，接触或抬升本身不算最终成功。原 `report.json` 的 `acceptance` 字段仍带旧工具的每类 70%、总体 75% 参考阈值，不代表本轮用户指定的验收门槛。没有验证集，也没有“最佳模型”选择。

## 准备检查与限制

`cpu_verified.json` 记录八组真实训练样本的原图传输、保存处理器输出与动作反归一化往返。`tests.log`、`scene_arguments.log`、`verification.json` 保存本次接口测试与启动参数检查；这些不能代替 GPU 重载和独立闭环评估。

本次环境 `torch.cuda.is_available()` 为 false，`nvidia-smi` 无法连接驱动。`final_bridge_verified.json` 只有手动 GPU 核验成功后才会生成，本次不伪造通过标记。

共享 ROS 代码、冻结源清单、训练数据和准备处理器由 `source_snapshot.json` 固定；不修改现有训练目录。若相关文件发生变化，接口会拒绝启动，应核对变更原因后重新检查接口，不能直接删除保护条件。
