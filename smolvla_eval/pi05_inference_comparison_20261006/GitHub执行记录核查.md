# GitHub 执行记录核查（2026-10-06）

## 结论

查到公开部署代码和实机视频，但未找到 YAM 与 ForceSight 对应模型的完整逐步执行日志。不能从预测 horizon 或采集 fps 推断每次执行帧数。

| 来源 | 可核实内容 | 未核实内容 |
|---|---|---|
| npow/openpi YAM，固定提交 8539331ec241dfa3b232fc792115548348567bc5 | README：15 Hz 数据/动作节奏，预测 horizon 15；关节 delta，夹爪 absolute；Servo 服务命令；evaluate_yam_open_loop.py 是留出集预测误差评估 | 匹配的硬件客户端、执行前缀长度、实机成功率、逐步执行日志 |
| TAP-VLA 项目 GitHub | index.html 提供纯 pi0.5 与 TAP-VLA 等四种方法、四任务的实机视频 | 完整机器人部署客户端、执行前缀、实时控制频率、逐步日志 |
| TAP-VLA 论文 | 约 10 Hz 记录、预测 horizon 10、7 维关节 delta 加二值夹爪、AdamW 20k steps / batch 32 | 每次是否执行全部 10 帧；HF 发布 checkpoint 是否与论文接口完全相同 |
| 官方 OpenPI DROID main.py | 15 Hz、默认每次执行 8 帧后重新推理、夹爪阈值 0.5 二值化、joint_velocity + gripper position；保存视频与用户打分 | 该代码本身不是某个 YAM/ForceSight checkpoint 的成功日志；代码仍断言 (10,8)，不可把它当成所有 pi0.5 配置的预测长度 |

ForceSight 模型卡的作者标识、四任务、四种方法及 20k checkpoint 与 TAP-VLA 高度吻合，因此将其视为相关研究证据；项目页没有明确链接该 HF checkpoint，仍不宣称精确版本等同。论文约 10 Hz 与此前读取的 HF 数据 15 Hz 不同，也不能混用。

## 原始链接

- YAM README：https://github.com/npow/openpi/blob/8539331ec241dfa3b232fc792115548348567bc5/examples/yam/README.md
- YAM 离线评估：https://github.com/npow/openpi/blob/8539331ec241dfa3b232fc792115548348567bc5/examples/yam/evaluate_yam_open_loop.py
- TAP-VLA 项目代码：https://github.com/tap-vla/tap-vla.github.io/blob/main/index.html
- TAP-VLA 论文：https://arxiv.org/html/2606.29089v1
- 纯 pi0.5 药瓶实机视频：https://tap-vla.github.io/static/videos/pi05_medicine.mp4
- TAP-VLA 满瓶实机视频：https://tap-vla.github.io/static/videos/tapvla_full.mp4
- 官方 DROID 客户端：https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py

## 与当前 Z1 的关系

当前模型预测 50 帧，SH 默认执行 5 帧，用户可覆盖执行 10 帧；动作时间间隔 10 Hz，发布插值 60 Hz。DROID 的 8/15≈0.53 秒可作为重规划时间尺度参照，不能证明当前 Z1 应执行 8 帧或夹爪应直接二值化。当前 Z1 是绝对关节位置与连续夹爪角度，动作定义也不同。
