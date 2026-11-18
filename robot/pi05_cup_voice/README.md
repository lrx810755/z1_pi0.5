# π0.5 杯子语音评估

启动：`bash /home/lp531/SH/start_pi05_cup_voice.sh`。如果麦克风选择不正确，追加 `--mic-index 编号`；用 robot 环境的 `asr.py --list-mics` 查看编号。

模型固定为 `/home/lp531/smolvla_training/pi05_cup_recovery513_lora_20261029/checkpoints/240000/pretrained_model`，adapter SHA256 为 `45237da918a747e15b4491ab593882e321db363c70014d29ba1337f2b3e23d67`。10Hz，预测30步、执行15步，底层插值60Hz。杯子使用原场景80g物理配置，非300g变体。独立 ROS_DOMAIN_ID=123，场景端口8826。

在终端输入 `start`，等待场景随机化完成，再说“小白抓取杯子”；每次尝试结束后再次输入 `start`。执行时可说“停止”请求停止，Ctrl+C退出并清理本入口启动的进程。只允许杯子目标，其他物体会被后端拒绝。Whisper small 本地识别，默认使用规则解析，不加载Ollama。

输出：`/home/lp531/smolvla_eval/pi05_cup_voice/pi05_cup_voice_时间/episodes/`，保留result.json、trace.jsonl、prediction_chunks.jsonl、input_sync.jsonl及三路视频。日志 `/home/lp531/z1_queue_logs/pi05_cup_voice/`。

实现复用20261030 execute15评估的隔离runtime；语音后端修复随机化调用缺少目标参数并限制杯子。原ROS包和旧SmolVLA语音入口不修改。独立前端也限制杯子指令，避免非杯指令造成等待超时；停止事件在进入执行状态前清空，保留进入执行后的停止请求。

验证：6项CPU集成测试通过（真实文本指令解析、模型/执行参数、拒绝非杯子目标、停止、模拟场景随机化/抓取/结果状态流），shell语法检查通过。已验证训练占用时入口退出，未启动仿真或新GPU推理。真实麦克风、Whisper转录与仿真抓取端到端尚未验证；当前训练完成后进行。

麦克风：默认按名称选择 AB13X USB Audio 的输入设备，不依赖可变编号；可用 `--mic-index` 明确覆盖。系统默认输入也已设置为AB13X mono-fallback。
