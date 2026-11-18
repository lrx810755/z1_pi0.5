import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'summary.json').read_text());assert s['completed'] and s['trials']==12
stages=['home','motion_start','approach25','approach50','approach75','preclose','close','lift','prerelease','release']
models=['large160_080000','small4_008000'];lookup={(x['model'],x['stage']):x for x in s['offline']}
fig,ax=plt.subplots(figsize=(12,4.5));xx=np.arange(len(stages))
for j,m in enumerate(models):ax.bar(xx+(j-.5)*.36,[lookup[m,stage]['first_action_mm'] for stage in stages],width=.36,label=('160 episodes / 10000 updates' if j==0 else '4 episodes / 1000 updates'))
ax.set_xticks(xx,stages,rotation=30,ha='right');ax.set_ylabel('First-action position error vs demonstration (mm)');ax.set_title('Paired offline check: same 40 training observations');ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(OUT/'offline_comparison.png',dpi=160);plt.close(fig)
with (OUT/'autonomous_results.csv').open('w') as f:
 fields=['episode','repeat','success','failure','sim_s','physical_grasp','first_close_sim_s','ever_in_position_envelope','min_distance_to_position_envelope_mm','predictions','prediction_closes_in15','sensor_max_skew_s','sensor_max_age_s']
 w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(s['episodes'])
lines=['# 四轨迹 PI0.5 小数据模型检查与全自主测试','',
'## 结果','',
'最终 008000 checkpoint（1,000 次优化器更新）在 4 个训练场景上各测试 3 次：**0/12 成功，10 次夹爪碰桌停止，2 次 60 仿真秒超时，0 次有效物理抓取。** 这是训练场景记忆能力测试，不是独立验证集测试。','',
'## 测试条件','',
'- 使用全部 4 条小数据训练场景：原始 episode_000157、000154、000160、000155。恢复各自原 seed、网格和所有物体布局，从初始机械臂状态开始。',
'- 实时三相机＋7 维实际关节状态＋原任务文本。模型全程决定机械臂与夹爪动作，没有 MoveIt 接管、示范轨迹引导、提前提取未来闭爪信号或固定瓶子。',
'- chunk=15，每次执行前 5 帧再重新推理；标称动作 10 Hz、插值执行 60 Hz；保留当前关节范围和 0.1 rad 限步；接触等待关闭，已包含此前的二值夹爪阶段判断修复。',
'- 三次重复均使用 inference seed=123，测试物理执行重复性；不是三个不同采样种子。保存完整 15 帧预测、实际模型输入、逐命令轨迹、同步信息和三路录像。',
'- 推理期间仍使用现有同步执行流程，名义 10 Hz 不代表算上推理停顿后全程持续 10 Hz。输入归档中位耗时约 5.5–6.7 ms，推理中位耗时约 298–313 ms。','',
'## 逐次结果','',
'| 场景 | 重复 | 结束时间（仿真秒） | 结果 | 距位置抓取区最近距离 |','|---|---:|---:|---|---:|']
for x in s['episodes']:
 reason='碰桌' if x['stop_reason']=='finger_table_contact' else '超时未抓取'
 lines.append(f"| {x['episode']} | {x['repeat']} | {x['sim_s']:.2f} | {reason} | {x['min_distance_to_position_envelope_mm']:.1f} mm |")
lines+=['','最近距离是目标在夹爪局部坐标系中到评估器位置区间的距离，只反映位置，不包含方向或接触条件。12 次均未进入该位置区间。一次记录出现物体离桌标志，但没有双侧接触/有效物理抓取，不能计为抓起成功。','',
'## 闭爪问题','',
f"共记录 **{sum(x['predictions'] for x in s['episodes'])} 次实时推理**。所有完整 15 帧中的夹爪值均未达到闭爪阈值 0.5，因此本轮不是闭爪信号位于第 6–15 帧而被执行窗口遗漏。",'',
'相反，离线输入四条示范的真实闭爪前画面，小模型都能在前 5 帧内给出闭爪信号。说明当前自主失败发生在接近阶段：模型没有把系统带到这些正确的抓取状态。它是否已学会后续搬运，不能由本轮自主结果确认。','',
'## 同输入离线对比','',
'让上一轮 160 条模型和本轮 4 条模型分别处理同样的 40 个真实示范输入。下面为第一帧预测与对应示范目标的夹爪参考点位置误差，由 URDF 正运动学计算，尚未执行限位、限步。','',
'| 阶段 | 160 条模型 | 4 条模型 |','|---|---:|---:|']
cn={'home':'初始','motion_start':'开始运动','approach25':'接近 25%','approach50':'接近 50%','approach75':'接近 75%','preclose':'闭爪前','close':'闭爪起点','lift':'闭爪后 20 帧','prerelease':'释放前','release':'释放后 5 帧'}
for stage in stages:lines.append(f"| {cn[stage]} | {lookup[models[0],stage]['first_action_mm']:.2f} mm | {lookup[models[1],stage]['first_action_mm']:.2f} mm |")
lines+=['','初始位置误差下降，但接近、闭爪前和搬运相关精度没有整体改善。小模型四个闭爪前样本的首帧误差分别约 7.43、8.66、22.31、8.18 mm。两个模型的 4 个初始输入都仍出现了原始预测关节越界。','',
'这不是严格的数据规模因果对照：大模型训练 10,000 次更新，小模型训练 1,000 次，且小模型的学习率日程按短训练自动压缩。只能说明本次小数据配置尚未学会这些场景，不能据此断言小数据必然无效或 PI0.5 无法完成任务。','',
'## 输入与执行核对','',
f"- 所有物体恢复位置的最大误差约 {max(x['max_layout_error_mm'] for x in s['episodes']):.3f} mm。",
f"- 传感器配对最大时间差约 {max(x['sensor_max_skew_s'] for x in s['episodes'])*1000:.1f} ms，取样时最老数据约 {max(x['sensor_max_age_s'] for x in s['episodes'])*1000:.1f} ms；没有同步超时或空白相机中止。该数据年龄不包含后续推理耗时。",
'- 恢复后的首次实时图像与原始录制图像，RGB 平均绝对像素差约 0.67–1.98（0–255 尺度），未发现明显视角或颜色通道改变。',
'- 又用实际 LeRobot 视频读取器核对四条轨迹的初始/闭爪前帧，解码图像与原始 JPEG 平均差约 1.41–1.65，状态最大差约 1.05e-7；未发现离线检查拿错帧或明显训练视频错配。',
'- 本轮预测没有 joint5 越界，不能直接沿用上一轮“joint5 被裁到 77 度导致下压”的解释。',
'- 4 次运行的部分原始预测已经进入几何低高度风险区，且这些预测的关节裁剪没有改变目标；说明至少这些危险目标在执行前就存在。其余接触不能仅凭简化 URDF 网格高度解释，实际停止以 PhysX 接触为准。',
'- 12 次均未固定瓶子。测试结束已停止策略控制并关闭推理进程，仿真保持运行。','',
'## 判断','',
'本次 4 条/1,000 次更新尚未达到“把训练场景学会”的目标。低训练 loss 不代表到位动作已经准确；现有证据仍把主要问题指向自主接近时的预测质量与闭环偏离，不能用扩大执行窗口来解释本轮未闭爪。后续应先在这份固定小数据上改善拟合并按动作误差和全自主结果验收，不能因为这次短训练失败就直接决定重录更多数据。此次没有启动追加训练。','',
'## 文件','',
'- summary.json、autonomous_results.csv：12 次自主结果。',
'- offline_predictions.jsonl、offline_summary.csv、offline_comparison.png：80 次同输入离线推理及对比。',
'- repeat1/、repeat2/、repeat3/：逐次预测、输入、轨迹、三路录像。',
'- initial_image_comparison.json / .jpg、training_video_comparison.json：输入核对。',
'- geometry_summary.json：原始/裁剪目标的几何风险检查。',
'- first_round_endframes.jpg：第一轮四场景的结束画面。']
(OUT/'实验报告.md').write_text('\n'.join(lines)+'\n')
print('Report and plot saved; verified 12 trials and',len([json.loads(x) for x in (OUT/'offline_predictions.jsonl').read_text().splitlines()]),'offline inferences.')
