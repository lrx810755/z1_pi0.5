# Renderfix500 ACT：独立解码与训练入口

本目录只读取新数据，不修改采集器、生产 LeRobot 或旧训练目录。当前源记录是三路 JPG 序列，并非压缩视频；入口会逐帧完整解码三路原图、验证原分辨率及 JPEG 内容 SHA256，保存与旧部署预处理逐像素一致的224×224 float32裁剪缓存。保留所有原始 JPG，不重复存储原分辨率解码数组。

```bash
/home/lp531/SH/start_act_cup_renderfix500_20261027.sh
```

入口依次执行：500条完整性门禁 → 冻结元数据/空间拆分/统计 → 全量解码 → 缓存哈希及原图像素配对审计 → 真实训练批次4步显存测试 → 检查点重新加载及 heldout loss → 全新 ACT训练90,000步 → 最终检查点重新加载验证。默认两个CPU解码进程。必须有且仅有400 basic +100 accepted correction，类别为approach20、shifted_target20、close_transition15、placement15、low_clearance10、regrasp10、transport10；plan_hash必须为6640b372e672501d989500e69aa5fc676c46a47a9ffd6f9f4af3f71aa440f748。未满500时立即退出，不产生缓存。

可单独审计、准备或解码：

```bash
/home/lp531/pi05_lora_workspace/.venv/bin/python inventory.py
/home/lp531/SH/start_act_cup_renderfix500_20261027.sh --prepare-only
/home/lp531/SH/start_act_cup_renderfix500_20261027.sh --cache-only
```

运行 Python3.12 的 `/home/lp531/pi05_lora_workspace/.venv/bin/python` 并固定 `PYTHONPATH=/home/lp531/pi05_lora_workspace/lerobot/src`。`unifolm-vla`是Python3.10，无法导入当前LeRobot的typing.Self；不用它执行ACT。其numpy可以运行纯数据规则测试。

训练建议已保存到 `training_proposal.json`：batch32先做4步真实数据显存测试，CUDA OOM时保存失败日志并改为16；90k优化更新；学习率和backbone学习率均1e-5、weight_decay1e-4、clip1、FP32、不累积；chunk16、n_action_steps5；全新ACT，只有ResNet18的ImageNet初始化，不恢复旧杯子模型。每5k步做验证/保存；第20步额外保存第一个可审阅检查点。每个验证episode固定抽取最多32个合法窗口，总计最多1600；保存验证loss历史及最佳验证loss检查点指针。最终验证loss只是离线指标，不能替代物理任务成功率评估。

固定保留格点(1,1)、(2,3)用于验证，目标450训练/50验证。纠正片段引用的basic必须同格点同seed；允许同split共享参考场景，禁止跨split seed或帧元数据内容重叠。Basic使用已审计过的单次close/release标签规则；纠正片段只使用保存的`gripper_labels.json`，完整保留多次close/open/regrasp。逐帧排除`expert_frame_mask=false`，同时要求整个16-action窗口都在expert mask内，并与保存的`valid_action_horizon16_anchors`逐项一致；尾部不足16步不训练也不验证。归一化统计只使用训练episode的expert帧，保留state里的真实连续夹爪位置。

抽样概率质量严格为basic75%、correction25%。每个来源内部，所有合法窗口anchor初始权重1；距离任一夹爪intent切换前后8帧的anchor权重1.5，再分别归一化到75/25。没有验证帧或干预窗口进入训练采样。不按纠正类别再次均衡。当前494条为141272帧、118510个训练合法窗口，90k×32=288万anchor抽样，约24.3次窗口等价遍历；最后6条完成后会重新计算精确数量。相比旧236条的53235训练窗口，当前窗口数约2.23倍，采用90k步为更大数据提供更多覆盖。batch16回退时共144万anchor，约12次遍历，配置会明确记录。

当前494条裁剪缓存约237.7GiB，最终约240GiB；启动时检查剩余空间足够缓存和额外80GiB检查点余量。单episode、单camera mmap及8帧解码块，训练worker仅保留24个mmap句柄，避免把全部图像载入RAM。原始全分辨率图片继续从源目录读取。

`workflow_status.json`、`prepare_progress.json`、`cache_progress.json`、`cache_workers/*.json`、`cache_verify_progress.json`、`live_training.json`可查看进度；各阶段也有独立日志。`source_pins/`冻结帧/summary/纠正标签/掩码/物理证据和plan；每个camera缓存旁的JSON记录每张源图SHA和缓存SHA；`runtime_provenance.json`、`code_pins/`固定解释器、依赖版本、训练源码及ImageNet权重哈希。已完成缓存支持安全重验后复用，不复用任何旧训练的图像缓存。

`first_update_verified.json`记录真实optimizer step=1、有限loss和action head确实变化；`sampler_verified.json`记录采样断点重放；`checkpoint_reload_verified.json`/`final_checkpoint_reload_verified.json`记录三路原图预处理、动作归一化往返、有限heldout loss和16×7输出；`best_validation_checkpoint.json`指向离线loss最佳保存点。新训练输出全部在本目录`output/`，显存测试在`smoke_b32/`或`smoke_b16/`。已有同名训练输出时拒绝覆盖；长训练中断后需要检查现有检查点再明确设计恢复，不会自动重启或自动恢复旧模型。

验证命令（只做CPU/合成数据测试，不启动GPU训练）：

```bash
PYTHONPATH=/home/lp531/pi05_lora_workspace/lerobot/src \
 /home/lp531/pi05_lora_workspace/.venv/bin/python -m unittest discover -s . -v
```

必需源元数据文件逐个强制存在且非空，不再跳过缺失证据。纠偏physics_status逐行解析，样本数必须严格等于dynamic_samples；每条必须frozen=False、target_collisions_enabled=True、两侧finger/table contact=False，并要求有限、单调不减的simulation_time。损坏JSON、缺字段、空流或样本数不符均阻断准备。只读inventory也使用相同证据门禁。


## 实际执行（2026-10-27）

17:50完成100条纠偏独立审核，400+100合计500条、142877帧（三相机428631张JPEG）。450训练/50验证；119776个训练合法16步窗口、12703个验证合法窗口；2397个干预帧及相交窗口排除。原始JPEG保留，仅缓存与部署完全一致的224×224 float32裁剪，258083868672字节，约240.4GiB。

17:52根据实测内存将解码进程从2增加至4，已完成缓存逐份核对后复用；训练worker仍2。实际进程见workflow_launch.json；进度见workflow_status.json、cache_progress.json、cache_verify_progress.json、smoke_live.json、live_training.json；最终实际参数见selected_hyperparameters.json。批量32经真实4次更新实测后确定，CUDA显存不足则16；全新ACT初始化，ImageNet ResNet18骨干，绝不恢复旧236条模型。

纠偏原始episode_000055因物理日志585行/元数据586已隔离保留于采集日志目录rejected_sources，任务042重新采集为episode_000097。合格纠偏目录episode_000055的编号缺口是有意保留，不影响500条冻结索引。采集审核证据：/home/lp531/z1_queue_logs/cup_correction100_full_renderfix_20261027/completion_verified.json。

18:08全量缓存校验通过：1500文件SHA、2000个窗口原图/缓存像素误差0，动作与状态一致；batch32四次真实更新峰值显存3.03GiB、有限loss、动作头权重确实更新；测试检查点在基础与纠偏独立验证示范上重载和反归一化通过。18:08:32正式90000步ACT训练已启动，实际batch32、LR1e-5、FP32；已生成第20步正式检查点。训练尚未完成，实时进度以live_training.json为准。
