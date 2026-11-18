# Z1 Isaac Sim 抓取抬升现存问题记录

更新时间：2026-07-28  
适用工程：

- `/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge`
- `/home/lp531/isaac_assets/z1/scripts/run_z1_ros2_scene.py`

## 1. 当前结论

当前系统已经完成“物理夹取优先、验证后才能 ATTACH”的闭环：

- 闭爪和试抬期间苹果始终承受完整重力，不再使用重力补偿；
- `attached=false`、`transport_constraint_active=false` 的试抬阶段共连续记录
  45 帧；
- 顶部连续 5 个样本均通过双指接触、离桌、动态物理、严格包络、最小抬升和
  相对跟随检查后，才允许授权 `ATTACH`；
- Isaac 场景在授权时和真正绑定前分别独立复核物理证据，无法通过客户端命令
  提前吸附；
- 2026-07-28 完整回归成功保存为 `episode_000003`，frame 297 首次确认
  `physical_grasp_ok=true`，frame 303 才首次进入运输绑定。

旧 `episode_000002` 的试抬阶段使用过重力补偿，不能再作为完整重力物理夹取的
训练基准；应排除或明确标记为旧版辅助验证数据。

## 2. 已经确认正常的部分

以下部分目前没有证据表明是主要故障源：

1. MoveIt 可以生成到达最终预抓取位和试抬位置的轨迹。
2. 机械臂能够执行约 40 mm 的慢速试抬动作。
3. 试抬开始前场景状态为：
   - `attached=false`
   - `transport_constraint_active=false`
   - 目标为动态刚体
   - 目标未冻结。
4. 未通过物理试抬时，Isaac 会拒绝建立辅助搬运约束。
5. 失败回合的录制会被丢弃，不会进入 SmallVLA 正式训练集；场景保留在失败
   位置，由操作者点击“一键重置 Episode”后才归位。
6. 抓取成功判定继续要求：
   - 夹爪受阻；
   - 物体位于统一双指包络内；
   - 物体至少上升 25 mm；
   - 相对平移漂移不超过 15 mm；
   - 相对旋转漂移不超过 15°；
   - 物体离开桌面；
   - 未建立任何辅助约束；
   - 接触报告可用时，左右指持续接触同一个目标。

## 3. 主要问题表现

当前失败过程可以概括为：

```text
机械臂到达最终预抓取位置
→ 苹果解冻并稳定
→ 夹爪低速闭合并出现关节受阻
→ 机械臂开始无附着试抬
→ 苹果只被带起约 4 mm，随后滑动、滚动或继续由桌面承重
→ 未达到 25 mm 上升门槛
→ physical_grasp_ok 保持 false
```

这说明“夹爪运动受阻”不等于“已经形成可承重的双指夹持”。

## 4. 已确认的几何与物理限制

### 4.1 苹果碰撞代理过于理想化

当前配置保留高精度扫描网格作为视觉模型，但动态碰撞使用半径 23 mm 的不可见
球体代理。

球形代理存在以下问题：

- 与两指容易形成面积很小的点接触；
- 接触点稍微偏高或偏低就会产生滚动力矩；
- 固定指与活动指的接触可能不对称；
- 23 mm 可能导致闭爪后的有效法向力不足；
- 尝试增大到 24 mm 时，固定指第一次 1 mm 就位就会把苹果推出安全包络。

因此不能继续只靠盲调球体半径解决问题。下一版本应优先根据实际苹果尺寸制作
椭球体、低面数凸包或少量凸包组合。

### 4.2 单侧活动夹爪容易产生不对称接触

当前夹爪由固定指 `gripperStator` 和活动指 `gripperMover` 组成。活动指闭合时
会把苹果推向固定指。如果苹果中心存在几毫米误差，可能出现：

- 活动指持续推动苹果；
- 固定指只发生边缘接触；
- 单指顶住苹果；
- 两指虽然都接触，但接触高度不对称；
- 苹果在试抬时沿固定指滚动。

物体几何中心进入抓取包络，只能证明位置大致正确，不能单独证明已经形成对称、
持续、可承重的双指接触。

### 4.3 位置控制不能直接保证夹持力

当前夹爪参数为：

```text
drive_stiffness = 500
drive_damping = 50
max_effort = 30
close_duration = 4.0 s
contact_hold_preload = 0.05 rad
```

当前受阻判定使用 0.25 s 实际关节角窗口，窗口内运动不超过 0.006 rad 时近似认为
夹爪受阻。检测到受阻后，以实际角度为基础最多继续施加 0.05 rad 预载，不再强制
运动到完全闭合位置。

该策略能够减少高刚度挤压和苹果弹飞，但仍存在：

- 预载不足时，摩擦力不足以抵抗重力和抬升加速度；
- 预载过大时，球形苹果会被挤出两指之间；
- 关节角平台不是严格接触力测量；
- PhysX 接触震荡会造成实际法向力变化；
- 当前接口尚未提供直接、有限的夹持力或关节力矩控制。

### 4.4 摩擦系数不是唯一条件

即使已经使用高摩擦物理材质，最大可用摩擦力仍取决于双指法向力和接触位置。
如果接触点不对称、太靠近水果顶部/底部，或者苹果仍由桌面承重，提高摩擦系数
也不能保证抓取成功。

## 5. 新增 PhysX 接触报告状态

场景已经在以下刚体上启用 `PhysxContactReportAPI`：

- `gripperStator`
- `gripperMover`
- Cup
- Apple
- Peach
- Mouse
- BeverageBottle

接触状态来源包含：

- `CONTACT_FOUND`
- `CONTACT_PERSIST`
- `CONTACT_LOST`
- 左指接触对象；
- 右指接触对象；
- 左右接触冲量；
- 目标与桌面的接触；
- 双指连续接触帧数。

只有左右指连续 10 个物理帧接触同一个目标时，才发布：

```text
bilateral_contact=true
```

当前已经完成苹果静置桌面的 ROS 实测：

```text
contact_report_available=true
object_touching_table=true
object_touching_table_estimated=false
left_finger_contact_object=null
right_finger_contact_object=null
bilateral_contact_consecutive_frames=0
```

该结果符合苹果在桌面、夹爪远离苹果的真实状态，证明接触报告和 ROS 状态链路
已经工作。

## 6. 尚未确认的关键问题

由于当前还没有完成启用接触报告后的可靠完整闭爪/试抬回归，以下问题暂时不能
直接下结论：

1. 闭爪完成时是否确实形成左右双指接触。
2. 左右接触是否为同一个目标。
3. 双指接触能否连续维持至少 10 个物理帧。
4. 左右接触冲量是否明显失衡。
5. 试抬开始后哪一侧先失去接触。
6. 苹果是发生向下滑动、绕固定指滚动，还是始终未完全离开桌面。
7. 0.05 rad 预载是偏小，还是已经因接触位置错误而将苹果推出。

这些问题必须通过新接入的接触对象、冲量、连续帧和物体轨迹共同判断，不能继续
只根据视觉或最终高度猜测。

## 7. 当前外部阻塞

当前机器出现 NVIDIA 内核驱动与用户态库版本不一致：

```text
NVML_ERROR_LIB_RM_VERSION_MISMATCH
CUDA error 804
```

Isaac Sim 可以进入 CPU 兼容模式并完成静态苹果桌面接触探测，但该环境不适合
继续进行可靠的完整自动抓取动力学回归。

应先修复 NVIDIA 驱动并重启机器，再进行闭爪和试抬参数标定。

## 8. 下一步排查顺序

1. 修复 NVIDIA 驱动/库版本不一致并重启。
2. 确认没有同时运行第二套 Isaac/ROS2 栈，避免 `/clock` 回跳和
   `TF_OLD_DATA`。
3. 启动完整系统：

   ```bash
   source /opt/ros/humble/setup.bash
   source /home/lp531/ros2_ws/install/setup.bash
   ros2 launch z1_isaac_bridge z1_isaac.launch.py
   ```

4. 在闭爪与试抬期间记录：

   ```bash
   ros2 topic echo --full-length /smolvla/scene_grasp_status
   ros2 topic echo --full-length /smolvla/status
   ros2 topic echo /smolvla/physical_grasp_ok
   ros2 topic echo /smolvla/transport_constraint_active
   ```

5. 根据记录将失败分类为：
   - 单指接触；
   - 双指接触对象不一致；
   - 双指冲量明显不平衡；
   - 双指接触未连续保持；
   - 试抬后单侧先丢失；
   - 苹果滚动；
   - 苹果滑落；
   - 苹果仍接触桌面。
6. 测量苹果真实三轴尺寸，制作椭球或少量凸包碰撞代理，替换当前 23 mm 球体。
7. 根据左右接触数据调整固定指就位量和有限预载。
8. 如果位置控制仍无法稳定夹持，再增加有限力矩或夹持力控制。
9. 严格无附着试抬通过后，才能回归辅助搬运、自然释放和 Episode 保存。

## 9. 最终通过标准

一次有效物理抓取必须同时满足：

```text
final_approach_complete = true
attached = false
transport_constraint_active = false
gripper_obstructed = true
inside_grasp_envelope = true
bilateral_contact = true
bilateral_contact_consecutive_frames >= 10
object_rise >= 0.025 m
relative_translation_drift <= 0.015 m
relative_rotation_drift <= 15 deg
object_touching_table = false
physical_grasp_ok = true
```

第一次物理抓取成功以后，才允许建立辅助搬运约束。不得通过降低 25 mm 上升
门槛、关闭碰撞、提前 attach 或强制物体跟随来制造成功结果。

## 10. GPU 恢复后的实测结论

GPU 阻塞已经解除。完整自动闭爪回归回答了第 6 节中的主要未知项：

1. 左右指能够同时接触同一个 `apple`。
2. 双指接触可持续，最佳达到 60 个物理帧，已超过 10 帧门槛。
3. 旧的固定指几何就位会把苹果推出包络；接触报告可用时必须跳过。
4. 冻结插入后的三轴追赶使用了错误的坐标假设，会放大误差甚至造成指尖碰桌，
   已禁用。
5. `0.07 rad` 有限预载能维持双指接触，但不能阻止当前球形苹果代理沿夹指
   接触面上爬。
6. 接触事件提供的冲量字段仍为 `0.0`，因此当前只能把接触对象和持续帧作为
   可靠观测，不能据此比较左右法向力。

最终代表性失败为：

```text
frozen_local=(0.142747,-0.001636,0.019376) m
settled_local≈(0.137089,-0.003624,0.022497) m
closed_local=(0.136847,-0.003685,0.022452) m
contact_angle=-0.644 rad
bilateral_contact_consecutive_frames=60
strict_z_max=0.022 m
z_excess=0.000452 m
```

系统因超出严格包络 0.452 mm 而拒绝，没有进入附着或录制。另一次已进入无附着
试抬的回归中，物体上升约 0 mm、夹爪跟踪误差约 36 mm，说明问题不只是成功
阈值边界，而是球形代理的接触法向和有限夹持能力仍不足。

## 11. 更新后的下一步

1. 测量苹果三轴尺寸，以椭球或少量凸包替换 23 mm 球形碰撞代理。
2. 保持严格抓取包络和 25 mm 无附着上升门槛不变。
3. 用真实接触对象、持续帧、物体局部轨迹验证新代理是否消除闭爪上爬。
4. 若新代理能稳定处于包络内但仍不能随夹爪上升，再增加可限力矩的夹持控制，
   而不是继续提高位置预载。
5. 接触冲量在 Isaac Sim 5.1 事件接口中仍为零，后续如需力平衡，应单独接入
   可验证的接触力/传感器通道。

## 12. 夹指—桌面三体卡死修复结果

已经实施以下组合：

1. 苹果闭爪姿态沿世界 `+Z` 抬高 4 mm。
2. 新增左右夹指碰桌真实接触状态，任一夹指碰桌立即停止闭爪。
3. 闭爪受阻后先保持当前角度；只有原始双指接触成立才允许有限预载。
4. 苹果由球形代理改为半轴 `(23.0,22.5,21.5) mm` 的单凸包椭球。
5. 持续双指接触后允许最多 1 s 动态稳定，但严格包络不变。

最终回归中左右夹指碰桌均为 `false`，持续双指接触达到 76 帧，闭爪稳定位置
位于严格包络内，并成功进入无附着试抬。苹果实际上升约 5 mm，但跟踪误差约
32 mm，仍未达到 25 mm 成功门槛。

因此原先观察到的夹指—桌面卡死已被保护并在代表性回归中消除；当前主要问题已
转移为试抬时夹持力不足或接触丢失。不得恢复球形代理、提前附着或降低上升门槛。

## 13. 夹爪为何不继续关闭及本轮修复

轨迹日志证明控制器没有停止发命令。闭爪受阻角约为 `-0.642 rad`，后续目标已
依次推进到 `-0.621、-0.601、-0.581、-0.561 rad`；实际关节不动或只移动很少，
是移动指受到苹果刚性碰撞负载、苹果又受到桌面约束造成的结果。

本轮将一次性预载改成四级有限预载，并在第一档期间执行 3 mm 世界 `+Z` 上抬
解卡。局部固定指方向的就位方案在实测中会撞桌，已撤销，保留真实碰桌立即停止
的保护。

代表性结果：

```text
initial_contact_angle≈-0.6422 rad
final_actual_angle=-0.6354 rad
continued_closure≈0.0068 rad
bilateral_contact_frames=161
object_rise=0.006 m
tracking_error=0.030 m
required_object_rise=0.025 m
```

所以“继续关闭”已生效，但物理夹持仍不足以通过试抬。下一步应检查夹指碰撞形状
与视觉模型的贴合，以及试抬期间左右接触何时丢失；不应继续盲目增大位置预载或
降低成功阈值。

## 14. 最终解决结果：严格抓取与蓝箱放置均成功

最终问题不是单纯“力矩不够”，而是四个因素叠加：

1. 固定指中央接触面过低，活动指先压住苹果后形成单指—桌面卡滞。
2. 固定指垫抬高后，冻结插入会提前碰撞，因此接近与夹持需要分阶段启用碰撞。
3. 过大的 `0.08 rad` 预载和较低线性阻尼会储存/释放接触能量，造成随机弹射。
4. 放置时夹爪在蓝箱内开爪会被箱壁挡住，虽已释放苹果，却差约 0.01 rad 才达到
   `released` 阈值。

最终组合为：

```text
apple pre-grasp center z = 0.008 m
stator pad local pose = (0.134, 0.0, -0.002) m
contact preload = 0.040 rad, step = 0.010 rad
max effort = 80
unjam lift = 0.010 m
physical test lift = 0.032 m at velocity scale 0.01
minimum object rise = 0.025 m  # 未降低
top hold = 0.50 s
apple damping = linear 10, angular 30
retention lip pose = (0.170, 0.0, 0.035) m
retention lip size = (0.004, 0.030, 0.015) m
```

止滑块只在双指闭合并完成 10 mm 离桌解卡后启用，释放/失败时禁用；严格试抬
期间苹果仍为动态刚体且 `attached=false`。只有试抬通过后才允许建立运输约束。

最终自动回归无需人工补动作：

```text
guided grasp: success
guided place: success
phase=SUCCESS
task_success=true
physical_grasp_ok=true
inside_success_volume=true
placement_stable=true
released=true
transport_constraint_active=false
transport_constraint_was_active=true
```

放置顺序最终固定为“箱内首次开爪 -> 上撤 40 mm -> 二次完全开爪 -> 稳定
判定”。因此苹果已经真实抓起并稳定放入蓝箱，本问题从“未解决”更新为“已解决”。
