# Z1 柱体/瓶体抓取：实机经验与 Sim 迁移记录

记录日期：2026-09-02
适用对象：Z1 六轴机械臂、Z1 夹爪、腕部 RealSense D435i、直立圆柱形瓶体
当前入口：`z1_examples/launch/red_ball_real_grasp.launch.py`
核心界面与规划逻辑：`z1_examples/z1_examples/red_target_gui.py`

## 1. 当前实机基线

当前代码中的有效默认值如下：

| 项目 | 当前值 | 坐标系/含义 |
|---|---:|---|
| 识别器 | `yolo` | 默认检测模式 |
| 标准类别 | `bottle` | 其他类别可由用户输入 |
| Z 位置补偿 | `-0.025 m` | `link00` 基座坐标系，向下 25 mm |
| Y 位置补偿 | `+0.020 m` | `link00` 基座坐标系，向 +Y 偏移 20 mm |
| 最终插入量 | `+0.030 m` | 沿目标四元数的夹爪局部 +X 轴深入 30 mm |
| 预接近距离 | `0.10 m` | 沿目标射线保留的预接近距离，不等于最终插入量 |
| 水平 pitch 目标 | `+5.0 deg` | 实机水平补偿值 |
| pitch 容差 | `+/-1.0 deg` | MoveIt 姿态约束局部 Y 轴容差 |
| yaw 容差 | `+/-5.0 deg` | MoveIt 姿态约束局部 Z 轴容差 |
| roll 约束 | 自由（`pi`） | 绕夹爪局部 +X 自转交给 joint6 约束 |
| joint6 目标 | `90.0 deg` | 独立关节约束，不得因四元数姿态调整而取消 |
| joint6 容差 | `+/-2.0 deg` | 规划后再次检查终点 joint6 |
| 夹爪张开目标 | `-0.90 rad` | `jointGripper` |
| 夹爪闭合选项 | `25.0 deg` | 程序转换为 Z1 夹爪关节的负角度符号 |
| 轨迹起点余量 | `0.03 rad` | 执行前实际关节与轨迹起点的最大容许误差 |
| 机械臂速度比例 | `0.05` | 接近与返回 HOME |
| HOME | joint1～joint6 均为 `0 rad` | 抓取后返回 HOME，再释放 |

参数定义位于 `red_target_gui.py` 顶部常量和 `parse_args()`；运行时也可以通过命令行参数覆盖。

## 2. 最终位姿构造顺序

当前接近抓取目标按以下顺序构造：

1. YOLO 检测 `bottle`，用彩色像素、对齐深度和 CameraInfo 得到相机坐标点。
2. 通过手眼标定与 TF 将瓶体中心转换到 `link00`。
3. 根据夹爪到瓶体中心的 XY 方向确定水平接近方向。
4. 将接近方向的 Z 分量清零，再施加 `+5 deg` pitch 实机补偿并生成 XYZW 四元数。
5. 对最终位置施加基座 `Z=-0.025 m` 补偿。
6. 对最终位置施加基座 `Y=+0.020 m` 补偿，避免夹爪 stator 与瓶体碰撞。
7. 沿目标四元数的夹爪局部 `+X` 深入 `0.030 m`。
8. MoveIt 同时施加位置约束、四元数姿态约束和 `joint6=90 deg +/-2 deg` 约束。
9. 规划最多尝试 5 次，选取最大关节变化较小的可行轨迹，并再次验证 joint6 终点。

关键原则：位置 offset、完整末端姿态和 joint6 约束必须同时满足；joint6 的 90 度原则不能用调整四元数代替或取消。

## 3. 水平姿态的实际结论

代码中的 `pitch=0 deg` 表示 URDF 中 `gripperStator` 的局部 +X 插入轴平行于 `link00` 的 XY 平面，并不保证实机夹爪外形肉眼绝对水平。

实机观察表明：将 pitch 从 `0 deg` 调整为 `+5 deg` 后，夹爪才接近实际水平。因此当前将 `+5 deg` 视为实机系统补偿量。可能来源包括：

- 实机关节零位与 URDF 理论零位的固定偏差；
- 夹爪安装轴、夹爪几何中心线与 `gripperStator +X` 不完全一致；
- 传动间隙、负载下垂或执行跟踪误差；
- 肉眼观察方向产生的透视误差。

Sim 迁移时不要直接假设仍需要 `+5 deg`。理想刚体仿真应先从 `0 deg` 验证：

- 如果 TF 计算出的夹爪局部 +X 已水平，Sim 应使用 `0 deg`；
- 如果仿真模型的夹爪网格/安装轴也存在相同几何偏差，再恢复 `+5 deg`；
- 实机与 Sim 的补偿值应分成两个参数，避免用一个常量同时标定两套系统。

水平度应通过 TF/四元数计算夹爪局部 +X 的仰角验证，而不是只看相机画面：

```text
tool_x = R(quaternion) * [1, 0, 0]
elevation = atan2(tool_x.z, hypot(tool_x.x, tool_x.y))
```

## 4. 距离相关经验

- 瓶体约在 `0.45 m` 时，实机抓取成功率较高，机械臂和瓶体间隙比较合适。
- 瓶体约在 `0.55 m` 时，更容易在夹爪闭合阶段把瓶体碰出夹持区域。
- 这不应只归因于夹爪闭合角。距离增加会放大深度噪声、手眼外参误差和像素中心误差；机械臂伸展后也更容易出现姿态/位置误差与结构下垂。
- 当前采用 `Y=+0.020 m` 避让 stator；迁移到 Sim 后应基于真实 collision geometry 重新验证，而不是无限增大 offset。

建议在 Sim 中至少建立 `0.45 m`、`0.50 m`、`0.55 m` 三个距离工况，每个工况改变瓶体横向位置并统计：

- 检测中心误差和深度误差；
- 目标位姿与实际末端位姿误差；
- stator、两指、腕部相机与瓶体/桌面的最小距离；
- 闭合后瓶体是否仍位于夹持区域；
- 规划成功率和轨迹执行成功率。

## 5. 一键抓取放置状态机

当前自动顺序为：

```text
approach -> grasp -> home -> release
```

- `approach`：识别瓶体并规划/执行带 offset、四元数和 joint6 约束的接近轨迹，同时保持夹爪张开。
- `grasp`：夹爪闭合到 25 度选项。
- `home`：joint1～joint6 返回 `0 rad`。
- `release`：在 HOME 位将夹爪恢复到 `-0.90 rad` 张开状态。

HOME 执行如果 MoveIt 返回超时，但机械臂已经连续满足位置和速度稳定阈值，会按安全到达 HOME 处理。

## 6. 相机与节点经验

`red_ball_real_grasp.launch.py` 会一起启动：

- Z1 实机 ros2_control、关节状态和轨迹控制器；
- MoveIt `move_group` 与 RViz；
- RealSense RGB、Depth、对齐深度；
- `red_target_gui.py`。

不要同时再启动另一份完整抓取 launch，否则可能重复启动相机、MoveIt或控制器。

单独运行 `red_target_gui.py` 只会订阅相机话题，不会启动 RealSense 发布节点。

已知现象：完成一次一键流程后，新建的临时 `RedTargetHandeyeApproach` 节点会再次执行 `wait_for_inputs()`，可能打印：

```text
Waiting for RGB image
```

GUI 是长期订阅者并缓存最后一帧，而每次规划创建的节点是新的独立订阅者。因此 GUI 有画面不等于新规划节点已经收到新 RGB；画面也可能是断流前的缓存帧。短暂等待后继续规划属于初始化等待，持续到超时则需检查 RealSense/USB 和话题发布率。

后续建议：规划直接使用 GUI 已冻结的检测数据，只等待规划必需的关节状态和 TF；GUI 同时记录最后一帧到达时间，超时后明确显示图像已停滞。

## 7. 迁移到 Sim 的参数分类

### 原样保留的策略约束

- 标准对象类别 `bottle`；
- 先对齐瓶体中心，再施加 offset；
- 夹爪局部 +X 为插入方向；
- `joint6=90 deg +/-2 deg`；
- pitch 约束容差 `+/-1 deg`；
- 一键流程 `approach -> grasp -> home -> release`；
- HOME 六轴均为 `0 rad`；
- 夹爪闭合 25 度的任务语义。

### 必须在 Sim 中重新验证的实机补偿

- `pitch=+5 deg`；
- `Z=-0.025 m`；
- `Y=+0.020 m`；
- 插入量 `0.030 m`；
- 夹爪关节角与实际指尖开口宽度的映射；
- 轨迹起点余量、速度、执行超时和 HOME 稳定阈值。

### 仿真模型必须对齐的基础项

- `link00`、`gripperStator`、相机 optical frame 的轴方向和 TF；
- 夹爪 stator、两指、相机和瓶体的 collision geometry；
- 瓶体尺寸、质量、质心、摩擦和接触参数；
- joint1～joint6 及 `jointGripper` 的零位、方向和限位；
- 相机内参、分辨率、深度单位、深度噪声和 RGB/Depth 对齐方式；
- Sim 发布的话题名与 GUI 默认话题一致，或者通过参数覆盖。

## 8. 推荐的 Sim 验证顺序

1. 不放瓶体，验证 HOME、joint6=90 度和夹爪开合方向。
2. 关闭执行，只发布目标姿态，验证 `gripperStator +X` 是插入方向。
3. 将 pitch 设为 `0 deg`，用 TF 数值检查局部 +X 仰角是否为 0。
4. 加入瓶体碰撞体，从零 offset 开始，逐项恢复 Z、Y 和插入量。
5. 检查预接近阶段、最终插入阶段以及闭合阶段的最小碰撞距离。
6. 在 0.45/0.50/0.55 m 三种距离做重复试验。
7. 最后加入相机噪声、深度空洞、控制延迟和关节误差，比较实机成功率。

每次只调整一个参数，并记录目标值、MoveIt 末端结果、TF 实际值、是否碰撞及抓取结果，避免位置补偿和姿态补偿互相掩盖。

## 9. 构建与重启说明

`z1_examples` 当前使用 `colcon build --symlink-install`，修改现有 Python 脚本中的参数通常不需要重新构建，但必须关闭并重启 GUI/launch，正在运行的 Python 进程不会自动加载新常量。

修改 C++、安装规则、包清单，或新增未被安装规则包含的资源时，才需要重新执行 `colcon build`。

## 10. 当前对应源码位置

- 默认参数：`z1_examples/z1_examples/red_target_gui.py:50`
- 水平四元数构造：`z1_examples/z1_examples/red_target_gui.py:72`
- Y offset 与局部 X 插入：`z1_examples/z1_examples/red_target_gui.py:1028`
- joint6 与姿态容差：`z1_examples/z1_examples/red_target_gui.py:910`
- MoveIt 位置/姿态/关节约束：`z1_examples/z1_examples/red_target_handeye_approach.py:525`
- 一键状态机：`z1_examples/z1_examples/red_target_gui.py:1391`
- 实机整套启动：`z1_examples/launch/red_ball_real_grasp.launch.py`
- Sim/实机工作单元基础启动：`z1_examples/launch/red_target_workcell.launch.py`
