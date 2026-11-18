# 相机时间同步失败：定位、修复与验证（2026-09-12）

## 结论与范围

本次 `no fresh image near ...` 的主要原因是 Fast DDS 默认共享内存段
容量不足，导致三路原始 RGB 图像在本机进程间传输时丢失。
平均话题频率不能揭示固定采样时间附近是否缺图；也不能仅凭 GPU 利用率
判断是否需要升级算力。

修复保持录制频率 10 Hz、图像匹配容差 45 ms、延迟 0.5 秒，以及
失败回合不可保存的校验不变。

## 排查证据

- 旧实例只有一个 `/clock` 发布者，独立订阅器没有发现图像时间戳倒退。
- 旧已保存回合的 `capture_time - simulation_time` 稳定在约
  0.50–0.55 秒，未发现随录制持续增长的积压。因而没有据此修改采样定时器。
- 旧实例的 35 秒独立订阅测试：wrist、global、vla_wrist 图像时间戳
  最大间隔分别为 133.3、283.3、150.0 ms，超过 90 ms 的缺口分别
  为 7、62、9 次。故缺口并非仅发生在录制器的选帧逻辑中。
- `/proc/<Isaac PID>/maps` 和 `/dev/shm/fastrtps_*` 显示使用默认
  512 KiB 段（文件含管理开销约 549408 字节）。
- 腕部单张 RGB 负载为 640×360×3 = 691200 字节；全局相机为
  640×480×3 = 921600 字节，均超过默认段容量。
- Fast DDS 官方说明：段容量接近或小于单条数据大小时存在高丢失风险。
  来源：[Shared Memory Transport](https://fast-dds.docs.eprosima.com/en/2.x/fastdds/transport/shared_memory/shared_memory.html)。

之前仅用 `ros2 topic hz` 得出的 107–142 ms 是墙钟到达间隔，不能直接
与仿真时间下的 45 ms 容差比较。本次依据消息头时间戳重新验证。
系统 UDP 接收溢出计数虽较大，但它是全机累计值，不能单独归因到本次相机。

## 同条件独立复现

`z1_isaac_bridge/test/camera_transport_smoke.py` 在 ROS_DOMAIN_ID=91
启动独立发送、接收进程，以相同尺寸发送三路各 300 张图像，避免访问机器人。

| 配置 | wrist 收到 | global 收到 | vla_wrist 收到 | 最大图像时间戳间隔 |
| --- | ---: | ---: | ---: | ---: |
| 默认共享内存 | 244/300 | 225/300 | 269/300 | 150 ms |
| 64 MiB 共享内存 | 300/300 | 300/300 | 300/300 | 16.7 ms |

默认配置测试退出码为 1；配置调整后退出码为 0。

## 实施

- 新增 `z1_isaac_bridge/config/fastdds_camera.xml`：64 MiB 共享内存段；
  保留 UDP 用于发现和远端节点通信。
- `z1_isaac.launch.py` 新增 `fastdds_profile` 参数，并在启动子进程前
  设置 `FASTRTPS_DEFAULT_PROFILES_FILE`。ROS 节点和 Isaac 都继承配置。
- `setup.py` 安装 XML；README 补充独立启动和诊断命令。
- 不修改系统全局网络配置，不放宽时间对齐标准，不复制旧图填补缺口。

已重启训练栈，并从进程环境和内存映射确认新配置实际生效。
旧循环停止时丢弃了未完成录制，保留 `episode_000001`、`episode_000002`。
恢复 `apple / easy / --episodes 98`，与既有 2 轮合计目标仍为 100 轮。

## 验证

- 构建成功：`PYTHONNOUSERSITE=1 colcon build --packages-select z1_isaac_bridge --symlink-install`。
  该环境中用户级 setuptools 与系统 colcon 不兼容，因此构建时隔离用户包。
- `test_recording_sync.py`、`test_episode_recorder.py`、
  `test_continuous_training.py`：14 项通过。运行时禁用无关 pytest 插件自动加载，
  避免系统 pytest 与用户级 anyio 插件版本冲突。
- 新实例 90 秒真实运动订阅测试：各路接收 3796、3797、3795 帧；
  三路图像时间戳最大间隔均为 33.3 ms，超过 90 ms 的缺口均为 0，
  图像时间戳倒退与仿真时钟倒退均为 0。
- `/clock` 发布者数量为 1。
- 追加 180 秒独立订阅测试：三路分别收到 7603、7605、7603 帧；
  最大图像时间戳间隔分别为 100.0、83.3、66.7 ms，wrist 有一次
  超过 90 ms 的缺口。此期间录制器日志没有同步错误。该结果说明
  不能把本次修复描述成“任意订阅者、任意时刻绝对零缺帧”。
- 最后追加 120 秒订阅测试：三路各收到 5196 帧，最大图像时间戳间隔
  均为 16.7 ms，没有超过 90 ms 的缺口或时钟倒退。
  虽然墙钟到达间隔曾达到约 189 ms，图像时间戳仍连续，进一步说明
  到达延迟不能等同于采集时间误差。
- 第 12 次尝试成功保存 `episode_000003`：410 帧、41 秒、零丢帧，
  `task_success=true`、`physical_grasp_ok=true`。
- 对该成功回合逐帧审计：三路图像相对采样时间的最大误差均约
  2.139 微秒；无重复或倒退图像；0.1 秒采样间隔最大数值误差约
  3.4e-14 秒。全部断言通过。这证明完整抓取、搬运、放置、保存流程
  在保持原同步标准的情况下可成功完成。
- 截至第 13 次尝试开始，录制器没有新增 `写入帧失败` 日志。
  修复后已新增成功 1 轮，加上原有 2 轮，累计 3/100；循环继续运行。
  其他失败原因是定位超差、夹持验证、MoveIt 错误；不应将这些失败
  归为相机时间对齐失败，也不应据此宣称整个抓取流程已稳定。

实际运行仍可能因抓取定位超差或夹持验证失败而自动丢弃回合。
这些失败应与相机同步错误分别统计；本次修复没有改变抓取验收标准。

## 复测方法

在包目录中 source ROS 环境后运行：

```bash
python3 test/camera_transport_smoke.py --profile config/fastdds_camera.xml
python3 test/camera_stamp_probe.py --duration 90
```

默认一键启动命令无需改变：

```bash
source /opt/ros/humble/setup.bash
source /home/lp531/ros2_ws/install/setup.bash
ros2 launch z1_isaac_bridge z1_isaac.launch.py
```

修复后训练日志：`/tmp/z1-continuous-sync-20260912.log`。
180 秒原始诊断结果：`/tmp/z1-camera-stamps-after-20260912.json`。
最后 120 秒诊断结果：`/tmp/z1-camera-stamps-final-20260912.json`。
本次实例的 ROS 日志：`/home/lp531/.ros/log/2026-09-12-01-48-15-429577-5314090-183684/`。
