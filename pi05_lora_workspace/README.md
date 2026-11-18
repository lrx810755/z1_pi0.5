# 瓶子 π0.5 LoRA 后台训练

数据：246 条完整轨迹 / 65,773 帧 / 10 Hz。训练 221 条、验证 25 条；验证按 baseline 19 / left 2 / right 2 / high 2 分层划分。所有中间帧保留，视频转换仅复制压缩包，不重新编码。状态和动作的归一化统计只使用训练轨迹。

## 配置

- 底座：官方 `lerobot/pi05_base`。
- LoRA：rank 32 / alpha 32，语言骨干与动作专家的 attention q/v 投影；动作输入、输出和时间投影完整训练；视觉编码器冻结。
- BF16、梯度检查点；micro batch 2 / 累积 4，有效 batch 8。显存预检失败时自动改为 micro batch 1 / 累积 8。
- 30,000 次优化器更新：batch 2 时共 120,000 micro steps；batch 1 时共 240,000 micro steps。学习率 1e-4，预热 1,000 次更新，余弦衰减至 1e-5，weight decay 0.01。
- 每 3,000 次更新保存并验证全部 25 条验证轨迹；完成后以验证损失最低的已保存检查点创建 `best` 链接。
- 动作仍为原始 7 维绝对关节命令；预测 50 帧，推理配置每次执行 5 帧。

验证损失用于比较检查点，抓取成功率需要后续仿真评估；更换底座本身不能保证解决腕部遮挡。

## 启动和文件

```bash
bash /home/lp531/pi05_lora_workspace/start_training.sh
```

任务：`pi05-bottle-lora-20261001.service`，由用户 systemd 托管，不依赖终端会话。
首次启动自动安装独立 Python 3.12 环境、下载底座、转换副本、验证数据、执行显存预检，然后训练。
启动脚本将当前终端的代理设置传给后台服务。权重固定为官方提交 `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`，优先官方源，失败后切换 `hf-mirror.com`，支持断点续传，完成后校验官方 SHA256。

- 准备日志：`/home/lp531/pi05_lora_workspace/background.log`
- 训练日志：`/home/lp531/pi05_lora_workspace/training.log`
- 输出：`/home/lp531/smolvla_training/pi05_bottle_lora_20261001`
- 分层清单：`split_manifest.json`
- 转换校验：`setup_verification.json`（转换成功后生成）
- 实际训练配置：`resolved_train_config.json`（训练启动时生成）

只提交后台任务，不持续监控。停止命令：

```bash
systemctl --user stop pi05-bottle-lora-20261001.service
```
