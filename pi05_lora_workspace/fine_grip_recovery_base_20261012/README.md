# π0.5 纠偏与闭爪细分主实验（2026-10-12）

本实验从 `/home/lp531/pi05_lora_workspace/weights/pi05_base` 新建 LoRA 和优化器，沿用上一轮 120,000 微步、梯度累积 8 的训练预算（15,000 次优化器更新），不是从旧 80k/120k adapter 继续训练。

- 数据：z1_bottle_with_fragments_20261012，211 条、47,135 帧，203 条训练 + 8 条验证。
- 微批次 1、累积 8、有效批次 8；bfloat16，梯度检查点。
- 学习率 5e-5，4000 微步预热，120000 微步余弦退火到 5e-6。
- LoRA r=32、alpha=32；沿用上一轮范围：语言模型与动作专家 attention q/k/v/o、FFN gate/up/down；action_in/out_proj、time_mlp_in/out 全参数训练；视觉编码器冻结。
- 动作块 15 步；三视角；绝对关节动作；夹爪 0/1。
- 原始/纠偏采样概率 70%/30%。实际记录的纠偏区间 2 倍；闭爪前 1.5、闭爪 2、准备抬升 2、初始抬升 1.5、稳定持握 1。重叠取最大值，最大 2。
- 已限定全部纠偏起点区间，并固定验证集；验证只从允许验证区间抽样。
- sampling_arrays.npz 和 window_sampler_impl.py 为本次运行冻结副本，不随外部配置文件后续修改而改变。

输出：`/home/lp531/smolvla_training/pi05_fine_grip_recovery_base_20261012`

后台 systemd 单元：`pi05-fine-grip-recovery-20261012.service`。

日志：本目录 training.log。首次在第 8 微步保存检查点，确认 adapter 参数实际变化及优化器 step=1；之后每 4000 微步保存和离线验证。

```sh
systemctl --user status pi05-fine-grip-recovery-20261012.service
cat /home/lp531/pi05_lora_workspace/fine_grip_recovery_base_20261012/live_training.json
```
