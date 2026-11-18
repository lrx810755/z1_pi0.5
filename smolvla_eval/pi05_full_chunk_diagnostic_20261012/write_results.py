from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parent;a=json.loads((P/'chunk_analysis.json').read_text());font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
im=Image.new('RGB',(1200,1100),'white');draw=ImageDraw.Draw(im)
for panel,d in enumerate(a['episodes']):
 rows=[json.loads(x) for x in (Path(d['path'])/'predictions.jsonl').read_text().splitlines()];x0=30+panel*400;y0=75;cellw=22;cellh=18
 draw.text((x0,5),f"{d['model']}\nseed {d['seed']} / {len(rows)} chunks",fill='black',font=font)
 for step in range(15):draw.text((x0+step*cellw,y0-22),str(step+1),fill='black',font=font)
 for j,r in enumerate(rows):
  for i,v in enumerate(r['gripper_raw']):
   z=max(0,min(1,v));color=(int(235-205*z),int(240-120*z),int(250-10*z))
   draw.rectangle((x0+i*cellw,y0+j*cellh,x0+(i+1)*cellw-1,y0+(j+1)*cellh-1),fill=color)
 draw.line((x0+10*cellw-1,y0,x0+10*cellw-1,y0+len(rows)*cellh),fill='red',width=3)
 draw.text((x0,990),'浅色：开爪；蓝色：闭爪\n红线右侧：末5步，未执行\n每行：一次完整15步预测',fill='black',font=font)
im.save(P/'full_chunk_gripper_heatmap.png')
s='''# 完整15步预测日志诊断结果

共完成3场、120次完整预测、1188个实际执行动作核对。模型、动作10Hz／插值60Hz、每轮执行前10步、60秒仿真超时及推理种子20261011均沿用原对照。使用原评估模块的诊断子类保存模型后处理后的完整动作块，原SH和生产代码未修改。

## 结果

|模型／场景|完整预测次数|前10步包含闭爪|仅末5步包含闭爪|15步全开爪|评估器结果|
|---|---:|---:|---:|---:|---|
'''
for d in a['episodes']:
 s+=f"|{d['model']} / {d['seed']}|{d['predictions']}|{d['prefix_close_chunks']}|{d['tail_only_close_chunks']}|{d['all_open_chunks']}|{'成功' if d['success'] else d['failure']}|\n"
s+='''
新120k远侧场景48次预测全部15步均为开爪，完整动作块夹爪值最大0.009608，远低于0.5。因此本次该场景不闭爪不能归因于把末5步闭爪截掉。

新120k近侧场景首次闭爪出现在编号10的预测块（编号从0开始），闭爪从块内第8步开始，已位于执行前缀。实际闭爪约13.53仿真秒，开爪约28.48秒。本次97个新模型动作块中，没有仅末5步包含闭爪的情况。

旧80k近侧场景编号8和9的预测块，均在第13步开始闭爪，被前10步截断；编号10预测块的闭爪前移到第8步并得到执行，最终评估成功。末段闭爪后移确实存在，但本次只观察到旧模型两次连续延后，随后完成转换，不能将此现象认定为新模型失败的主要原因。

以上第几步按1开始计数，JSON first_close_index按0开始计数。

## 新发现：行为与评估判定存在差异

新120k近侧视频显示瓶子被双指夹起、搬到箱上方并释放。闭爪期间物体根参考点最高约0.188385m；完整状态日志在多个时刻同时记录 bilateral_contact、lifted、holding_observed、relative_pose_stable 为真。但 physical_grasp_ok 始终为假，评估器将其归为grasp_failed，释放阶段跟踪也未确认。

不能据此宣布完整任务成功：最终瓶子倾倒／靠在箱边，最终根坐标(0.268175,0.088170,0.032595)m，不符合严格的端正放置要求。也不能将这次失败简单描述为没有夹取抬升。

源码显示物理抓持确认还要求接触几何有效、夹爪角度范围、瓶子倾斜不超过5度等条件；本次完整状态没有分别保存倾斜角及全部判据，尚不能确认哪一项阻止physical_grasp_ok。GraspPhaseTracker只有在先确认physical_grasp_ok后才标记release_started，因此前面的确认未成立时，即使模型实际开爪，评估释放序列也会保持未完成。这是判定链的限制，并非已证实的评估器错误。

## 可复现性及验证

相比上次相同模型／场景，3场初始物体最大位置差为1–3微米，最大初始关节差0.0002rad。推理种子相同，但新120k近侧从上次不闭爪变成本次夹起释放，远侧从上次空夹变成本次全程开爪。固定种子并不保证GPU物理仿真、运行时序与视觉反馈逐位一致；诊断保存日志也有少量额外开销。这次结果属于新的重复评估，不能倒推上次完整动作块内容。

全部120次预测的长度均为15。记录与input_sync预测编号逐项一致；1188个实际动作与对应完整块的前缀逐项一致。没有基础设施失败。诊断记录函数的两项测试验证了末段闭爪保存、开爪尾段区分、原动作不修改及追加记录。

## 对下一步训练的意义

当前证据支持保留“纠偏、闭爪转换和下降释放”的独立采样检查。没有证据支持为了新模型不闭爪而直接把执行步数从10改成15；这会同时改变反馈频率，需要单独对照。

在重新训练前，可对示范闭爪边界做离线完整15步预测，比较示范观测下能否预测转换；同时补齐physical_grasp_candidate各项判据，区分真实抓持问题与严格判定未通过。当前不能用这3场推断总体成功率，也不能将近侧新模型这次行为直接记为成功。

## 文件

每场episodes目录内的predictions.jsonl保存完整15×7动作、夹爪逐步值、执行／丢弃索引、输入同步编号、推理前完整接触状态、推理耗时及前后仿真时间。trace.jsonl和三视角视频沿用原格式。

- chunk_analysis.json：逐场统计、尾段事件及执行前缀校验。
- full_chunk_gripper_heatmap.png：全部120次预测的夹爪动作图。
- new120_near_lift_release.jpg：新模型近侧夹起、搬运和释放三视角截图。
- manifest.json：checkpoint路径、权重哈希、种子和配置。
- run_suite.py / run_diagnostic.sh / diagnostic_rollout.py：本次诊断入口，run_suite拒绝覆盖已有输出。
'''
(P/'15步预测分析.md').write_text(s)
print('report and heatmap written')
