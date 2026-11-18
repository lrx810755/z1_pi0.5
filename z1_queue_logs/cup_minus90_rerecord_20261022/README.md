# 杯子 −90° 示范重录

用户确认：先完成基础160条（保留124条原−90°，替换36条原+90°），后重录全部89条纠偏（其中31条原+90°），纠偏必须完整完成抓取、搬运、放置、松爪。

- 位置归档：`/home/lp531/smolvla_data/cup_minus90_rerecord_20261022/original_plus90_positions.json`，以及同名CSV。
- 全部替换任务：`collection_plan.json`。125个任务保留原种子、所有物体位置/姿态、方格、训练/验证划分、原始帧哈希和纠偏偏移参数。
- 新数据：`/home/lp531/smolvla_data/cup_minus90_rerecord_20261022/basic_replacements`、`corrections_full`。
- 基础集仅在36条全部通过后生成 `basic160/assembly_manifest.json`；124条原记录保留原始内容，36条映射到验收通过的新记录。episode目录及assembly manifest的编号为组合集权威编号，符号链接内的原始元数据保留各自录制时的编号。
- 原始数据、生产代码、原训练模型均保留；新配置只关闭杯子镜像roll候选。

## 验收

每次使用原种子和方格，完整复位，核对所有物体位置（2mm）和姿态（2度）。不自动换场景充数。原本post_shift_realign任务只在Home准备阶段复现原位移，并额外核对位移后的杯子位置。

首次闭爪前通过URDF正运动学测量末端roll，要求距−90°不超过3度。录制必须真实验证物理抓取、离桌、盒内稳定放置、release_sequence_complete、released且不再holding；不允许夹指碰桌。纠偏录制期间禁止冻结杯子/关闭其碰撞。夹爪标签必须含开→闭→开，并保留至少16帧释放尾部。缺失图像及同步问题由原录制器拦截。

纠偏沿用对应旧/新纠偏的偏离方向与幅度，围绕新的−90°参考抓取位生成偏离，沿预检路径恢复。恢复完成后使用生产完整抓取和放置流程，不再使用原短片段的6cm抬升及0.6s保持后结束逻辑。新纠偏同时有close_command_frame和open_command_frame；后续训练准备必须使用完整开闭标签，不能把尾部全部标成闭爪。

## 本次采集局部调整

初期原−90°位置重录暴露搬运/对中失败：搬运可在55mm误差内返回，但释放要求20mm，追加长直线对中有规划失败；有记录搬运后高度低于最低验收高度。

采集进程的独立assistant中将搬运XY目标收紧为最终释放标准20mm，将独立放置对中恢复为分段模式；目标高度增加12mm指令余量，并将原最多3次的闭环搬运修正增加到6次。物理高度验收、碰撞检查、夹持受力检查、盒内稳定释放验收保持生产标准。所有失败均丢弃，不计入配额。

## 执行与续录

`bash /home/lp531/start_cup_minus90_rerecord.sh`

ROS domain97、GUI port8799。程序先基础后纠偏，基于磁盘验收记录跳过完成任务。默认持续重试缺失任务直到全部完成；使用--no-until-complete可限制为三轮。每轮会记录缺失原位置，不自动改变原位置。全部125条通过后自动执行独立图像、标签、原场景和FK姿态审计，再标记completed。

`status.json`显示阶段和数量；`collection.log`为总日志；`runs/*/attempts.jsonl`记录每次结果，`runs/*/scene.log`保存仿真日志，`runs/*/collector_source`和`collector_manifest.json`保留本次运行的采集程序版本。预抓取单关节总行程筛选从2.7rad统一为搬运已用的3.0rad；关节实际限位及速度/碰撞约束保持。原始rerecord.json包含原路径、划分、姿态测量、原场景复现证明和最终物理状态。

针对最终位姿毫米级控制/动态残差，在闭爪前保持杯子动态，最多两次、每次最大12mm的低速位置校正（关节变化上限0.15rad）；不扩大闭爪位置包络，校正后继续生产原验收。超出12mm的偏差拒绝收入。

追加优化：在不改变原场景物体位置/姿态的条件下，增加−75至80度的接近方位候选（所有候选roll仍为−90°，碰撞约束仍生效）。带载搬运优先使用保持姿态的LIN直线路径，只有可恢复的规划失败才回退原OMPL自由规划；接触/运动保护失败禁止回退。对应8项协议/控制路由/持续续录测试已通过。

2026-10-22 19:42 状态检查发现末尾episode103重复失败500多轮。已终止无效循环，新增“连续三轮没有任何新增验收记录即needs_investigation退出”的进度保护，原位置不变。针对邻近瓶子与link04/link05的路径碰撞，开爪最终接近的LIN规划失败时允许同起终点的碰撞检查OMPL备用路径，并扩展全周方位角搜索（15度步长，roll仍为−90度）。10项测试通过。

## 2026-10-22 23:18 targeted recovery revision

Authorized by user: optimize and finish remaining corrections, keeping archived scenes,
minus-90 grasp and complete physical pick/place/release acceptance.

Stopped repeated unmodified collection (56/89). Pilot run 231522 isolated a phase
mismatch: post-shift setup was compared against the archived post-approach cup
position. Compare against original_correction.prepared_cup_shift.after instead,
with the unchanged 2mm tolerance. New001 subsequently passed complete recording;
old013 also passed in this pilot, giving 58/89 independently verified episodes.
New011 now passes setup-position validation but its approach planning still fails.

Added segment-specific planning evidence (*.planning.jsonl). New003 failed at
prepare_deviation; old004 continued moving the cup >5mm during preparation.
The next pilot (231849) evaluates side-entry clearance corridors and multiple
collision-checked minus-90 reference candidates. Requested deviation endpoint and
rotation remain unchanged; reference selection now requires complete recovery
planning, not only reachability of the final grasp. Per-reference errors and the
selected route are saved in *.reference.json. No collision/force/placement
acceptance thresholds were relaxed. Tests: 13 passing before pilot launch.

Z1_RERECORD_TASKS allows a comma-separated diagnostic subset; subset runs retain
full quota accounting and cannot claim dataset completion. Active process code
is snapshotted in its runtime; changes on disk apply only to subsequent launches
(except the original route modules, which must not be edited during a run).
