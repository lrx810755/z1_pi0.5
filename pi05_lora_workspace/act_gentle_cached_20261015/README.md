# ACT 温和采样与提前解码缓存，2026-10-15

用户授权：按提前解码缓存优化方法开始 ACT 重新训练，遵从此前温和采样规则。

数据仍使用 z1_bottle_with_fragments_20261012，共211条（160原始、32旧纠偏、19新片段）、47135帧。沿用 ACT 的190训练/21验证划分、仅训练数据计算的状态/动作统计及 ImageNet RGB统计。原数据、旧模型和评估SH默认模型不改动。

温和采样：原始70%、纠偏30%；各池内闭爪前2秒的开爪anchor与日志标记的闭爪前恢复阶段取并集，权重2，其余1。并集不叠加到4倍。只影响抽样概率，不加权loss。保持原ACT批准的采样区间，完整16步未来动作在区间内。重点区间当前总概率32.7114%，不是硬编码阶段比例。相较旧ACT唯一学习配置变化是上述anchor采样权重。

缓存：将9个源视频顺序解码；按视频时间戳与episode偏移对应到211条数据的每帧，保留PyAV RGB转换、负top补边到640正方形和torchvision带抗混叠缩放至224的处理。保存float32 RGB[0,1]内存映射数组，避免uint8重新量化带来的像素变化，纯图像数据约85.1GB。按camera/episode分片，原始视频不删除、不覆盖。

训练与验证从缓存加载，跳过一次已经完成的crop/resize；normalization仍由原处理器执行。保存到checkpoint的部署处理器仍包含原crop/resize，实际部署仍接收原尺寸相机图像。缓存、原读取路线的窗口和预处理数值需核对后才启动训练。

ACT重新初始化（ImageNet ResNet18骨干初始化、ACT主体新初始化），不续训旧50k。batch16，梯度累积1，50000次优化器更新，float32，chunk16，LR及骨干LR均1e-5，AdamW衰减1e-4，梯度裁剪1。固定seed20261013以便对照。缓存读取4个spawn worker，预取2，persistent workers开启，CUDA pinned memory由LeRobot配置开启。其余ACT配置沿用旧基线。

流水线 run_pipeline.py：等待缓存完成 → 18个窗口与原路径核对（3类来源各3条、每条2个时刻）、633个缓存文件尺寸与抽样值检查 → 4步smoke及真实参数更新 → 保存模型重载并验证3类真实验证集输入 → 正式50000更新。任何必要检查失败时停止，不继续正式训练。

输出模型：/home/lp531/smolvla_training/act_gentle_cached_chunk16_20261015。

状态：pipeline_status.json。训练日志：training.log。进度：live_training.json。首个真实参数更新：first_update_verified.json。采样核查：sampling_audit.json。缓存核查：cache_verified.json。最新checkpoint：latest_checkpoint.json。完整训练完成：completed.json。

已有后台流水线时不要重复执行 run_training.sh。

验证实测：CPU补边/缩放与缓存逐像素一致；原GPU部署路线归一化后最大差异2.094e-5（仅浮点实现差异）。18个窗口图像/状态/动作/padding对应关系及633个缓存文件检查通过。4个spawn worker短训练正常退出，保存checkpoint使用三类真实验证观测重载推理成功；正式训练首个参数更新和第20步checkpoint已经验证。读取样本对照耗时0.913秒（视频）与0.055秒（缓存），这只是局部读取测量，不代表完整训练固定加速倍数。
