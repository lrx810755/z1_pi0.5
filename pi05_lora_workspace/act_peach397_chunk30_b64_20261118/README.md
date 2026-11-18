# 桃子 ACT 独立训练

397 条冻结示范（340 基础、57 纠偏），117,537 帧；全部用于训练，无验证集。
原始数据为三路 JPEG 帧，共 352,611 张。全部核对冻结 SHA 后解码，按部署端相同的 float32 补边、裁剪、抗锯齿缩放至 224×224，用 zlib 无损压缩缓存（64.47 GiB）。不改写源数据。

ACT 预测 chunk 30，batch 64，FP32，梯度累积 1，60,000 次实际优化器更新。每批 48 基础、16 纠偏；普通/关键段/恢复段权重 1/1.5/3。7 维状态、7 维动作，使用本桃子数据 MEAN_STD。ResNet18 使用 ImageNet 初始化，ACT 其余部分新初始化，全部训练。

主体学习率 1e-5、骨干 1e-6，AdamW，weight decay 1e-4，梯度裁剪 1；1,000 步预热后余弦衰减至峰值的 0.1。模型维度 512、8 heads、FFN 3200、4 编码层/1 解码层、VAE 4 层/latent32、KL 权重 10。无图像增强。种子 20261118。

`start.sh` 执行一次完整流程：单元回归、CPU 图像与处理器核验、4 次 GPU 实际更新试训、保存模型重载、正式训练及最终重载。不自动重启。输出已存在会拒绝覆盖；不提供通用断点续训入口。

正式输出：`/home/lp531/smolvla_training/act_peach397_chunk30_b64_20261118`。
日志：`formal_training.log`。状态：`workflow_status.json`、`live_training.json`。每 100 步记录；第 100 步额外保存，后续每 5,000 步保存。

部署预处理器保留原始相机尺寸的补边裁剪与归一化；训练直接读取相同像素的缓存，跳过重复裁剪。配置为 `n_action_steps=1, temporal_ensemble_coeff=0.01`；未来闭环部署必须调用 `select_action` 才实际使用 temporal ensemble。本流程不驱动机器人，不执行闭环测评，训练损失和重载检查不能视为抓取成功率。

`code_snapshot.json` 固定本轮代码与相关 LeRobot 文件；`prepared.json`、`cache/completed.json`、`cpu_verified.json` 和试训重载证据绑定来源，正式启动前验证。
