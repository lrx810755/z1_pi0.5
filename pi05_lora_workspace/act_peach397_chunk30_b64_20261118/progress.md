# SDD ledger — plan: docs/superpowers/plans/2026-11-18-peach-act.md

Ruling: no Git repository at parent; isolate all new code and cache in dedicated ACT directory. No shared code edits.
Ruling: exact float32 cache is zlib-compressed to fit disk; sample estimate65GiB versus198GiB uncompressed. No lossy image quantization.
Pre-flight: Dataset consumes prepare arrays and indexed cache; sampler consumes prepared eligibility/weights; training consumes all and native processor serialization.

2026-11-18: 全部397条、117537帧、352611张源JPEG已完成SHA核对及实际解码，696.7秒，float32无损缓存64.4688GiB，磁盘剩余54GiB。
回归：7项测试通过（contracts.log），含真实CPU ACT原生优化器分组、采样比例与精确重放、专家窗口边界、损坏缓存、缓存容量和启动证据失效。
独立只读复核：review_peach_act确认源数据子集、掩码、统计与采样正确；发现的骨干学习率分组合并、部分缓存恢复容量预检、陈旧试训证据三个问题均已修复并有回归测试。
流程PID553926，CPU缓存与实际处理器比对进行中；未启动机器人或测评。不自动重启，正式训练需通过CPU与4次GPU实际更新及重载。

CPU核验通过：397条全部缓存1191个文件及索引SHA，1527个实际原始/缓存窗口逐像素相等、归一化往返一致、实际采样48/16与偏移77精确重放。
GPU试训4次batch64真实优化器更新通过，checkpoint在原始三路图像和已保存处理器上重载成功；30x7预测及temporal ensemble单步输出有限。
正式训练已从全新ACT初始化启动，PID554076；已核实300次真实更新，首次动作头参数已变化，主体/骨干学习率比10:1；第100步权重已落盘。目标60000次，后台继续，不持续监控，不驱动机器人。
