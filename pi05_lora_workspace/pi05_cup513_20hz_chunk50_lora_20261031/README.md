# PI0.5 LoRA：20Hz513条，训练chunk50

用户2026-10-31授权新一轮训练。数据来自已完成并最终验收的400基础＋100纠正＋13恢复20Hz示范，总311897帧；固定461训练、52验证，按原空间保留格划分。全部935691张源JPEG已计算SHA256，训练读取时核对文件属性，变化则重新核对内容哈希。源JSON和低维/采样/统计文件均冻结，原采集目录与旧模型不修改。

参数：同一官方pi05_base新建LoRA，rank32/alpha32/dropout0；语言与动作专家attention和FFN的LoRA，动作/时间投影全量训练，视觉编码器冻结，55249952个可训练参数。BF16，梯度检查点，batch1，梯度累积8，有效batch8。240000微步＝30000优化器更新；AdamW LR峰值5e-5、最低5e-6，betas0.9/0.95，weight decay0.01，grad clip1.0；warmup8000微步，余弦衰减。seed20261029与原轮保持一致。

训练chunk50@20Hz＝2.5秒预测范围，数据集动作张量[50,7]，完整窗口不得跨episode/干预；归一化仅来自训练episode专家帧。采样基础75%、纠正/恢复25%，恢复3倍/转换1.5倍/普通1倍，重叠取最大；转换半径16帧＝0.8秒保留原时间尺度，排除5006个非专家干预帧。图像三路原JPEG读取、相同letterbox裁剪、224×224；不开额外增强。

每8000微步（1000次更新）验证并保存，最多每验证episode16个窗口，记录最优验证检查点；正式训练第一更新另保存启动检查点。推理执行多少步与训练chunk分开，本任务不启动评估；旧配置n_action_steps5保留，后续使用predict_action_chunk的外部评估器需明确设置20Hz与执行窗口。

流程：CPU窗口/采样/种子分离/processor roundtrip→GPU8微步1次真实更新→checkpoint重载及[1,50,7]推理→全新正式训练→最终checkpoint重新加载验证。CUDA或严格基础权重加载失败、非有限loss等都会停止，避免随机权重或CPU回退。

状态workflow_status.json，实际微步/更新live_training.json，首更新first_update_verified.json；训练日志training.log；最优best_validation_checkpoint.json；训练与最终重载完成后workflow_status.stage才为completed。smoke与正式输出独立，正式不继承smoke更新。

模型输出：/home/lp531/smolvla_training/pi05_cup513_20hz_chunk50_lora_20261031
服务：pi05-cup513-20hz-chunk50-lora-20261031.service
