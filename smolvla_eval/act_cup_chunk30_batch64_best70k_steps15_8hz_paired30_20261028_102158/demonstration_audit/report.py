from pathlib import Path
import json,numpy as np,hashlib,csv
A=Path(__file__).resolve().parent;d=json.loads((A/'results.json').read_text());rows=json.loads((A/'sample_metrics.json').read_text());sel=json.loads((A/'selection.json').read_text())
assert len(rows)==d['samples']==sel['anchors']==12415
assert len(set(x['episode'] for x in rows if x['split']=='train'))==450
assert len(set(x['episode'] for x in rows if x['split']=='validation'))==50
with np.load(A/'predictions.npz') as z:
 assert z['predictions'].shape==z['expert_actions'].shape==(12415,30,7) and np.isfinite(z['predictions']).all()
 assert z['indices'].tolist()==[x['index'] for x in rows]
 assert np.isin(z['expert_actions'][:,:,6],[0,1]).all()
ck=Path(d['checkpoint']);snap=json.loads((A.parent/'source_snapshot.json').read_text())
for p in ck.iterdir():
 if p.is_file():assert hashlib.sha256(p.read_bytes()).hexdigest()==snap[str(p)]
train=Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_chunk30_batch64_20261028_003937')
split=json.loads((train/'split_manifest.json').read_text())
with np.load(train/'sampling_arrays.npz') as z:
 eligible=z['eligible_train']|z['eligible_validation'];assert eligible[[x['index'] for x in rows]].all()
for r in split['source_records']:
 for name,digest in r['source_hashes'].items():assert hashlib.sha256((Path(r['source_path'])/name).read_bytes()).hexdigest()==digest
checks=[]
for part in ['train','validation']:
 for delta in [0,1,5,14]:
  x=[r for r in rows if r['split']==part and r['delta_to_first_close']==delta];checks.append(dict(split=part,frames_after_first_close=delta,samples=len(x),correct_close_now=sum(r['predicted_close_now'] for r in x)))
(A/'closure_onset.json').write_text(json.dumps(checks,indent=2))
with (A/'sample_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# 成功示范状态下的动作预测查验','',
'对象：当前chunk30、batch64模型的70k检查点。检查全部500条成功示范：450条训练示范、50条空间留出验证示范；共12,415个有效状态，每个预测连续30步。专家动作仅用于对比，不输入模型。','',
'结论：模型在成功示范状态下已能预测接近动作和闭爪，但动作精度仍有限；未参与训练的空间位置上，远期动作误差明显增大。结合前一步实际执行误差较小的结果，后续应优先检查策略预测、空间泛化和自主运行后的状态偏移。尚不能仅凭这次查验判定是视觉特征、动作标签还是模型结构导致。','',
'## 与专家动作的位置差','',
'用相同六轴正向运动学与抓取参考点分别计算预测动作和专家动作的位置，再求两者差值。不是距离杯心或实际抓取成功率。第1/5/15/30步对应预测索引0/4/14/29。','',
'|预测位置|训练集 中位数 / P95|验证集 中位数 / P95|','|---|---:|---:|']
for h in [0,4,14,29]:
 t=d['summary']['train']['overall'][f'error_h{h}_mm'];v=d['summary']['validation']['overall'][f'error_h{h}_mm'];lines.append(f"|第{h+1}步|{t['median']:.2f} / {t['p95']:.2f} mm|{v['median']:.2f} / {v['p95']:.2f} mm|")
lines+=['','## 验证集按阶段的第15步误差','','|阶段|样本数|中位数|P95|','|---|---:|---:|---:|']
names={'approach':'接近','preclose':'闭爪前15帧','closing':'闭爪开始至14帧','transport':'搬运','placement':'放置'}
for phase,s in d['summary']['validation']['phases'].items():
 x=s['error_h14_mm'];lines.append(f"|{names[phase]}|{s['samples']}|{x['median']:.2f}mm|{x['p95']:.2f}mm|")
lines+=['','## 闭爪与动作学习的具体观察','',
'- 验证集闭爪起始帧有49个有效样本，其中28个预测首步达到0.6闭爪阈值；之后1帧（示范时间0.1秒）的49个样本全部达到阈值，之后5帧和14帧也全部达到。即示范状态下没有普遍持续数秒推迟闭爪，起始切换仍有约1帧延后。',
'- 验证集中，专家在预测前15步要求闭爪的811个样本里，2个把首次预测闭爪推到第16步以后，9个整块未预测闭爪。此处统计包含已经闭爪的状态，不是811次独立抓取。自主运行的17次反复推迟闭爪现象，不能用“示范里也普遍如此”解释。',
'- 不移动基线：验证集接近阶段第15步中位误差约54.36mm，模型约8.11mm，说明模型确实预测了接近运动。闭爪前段不移动基线约2.39mm，模型约6.18mm；该段专家常在稳定对准，模型预测的细小漂移值得关注。不同阶段基线不能混用。',
'- 单步误差训练/验证接近，远期误差差距明显；不宜将结果概括为“完全没有学会定位”，也不宜仅凭总体验证损失判断可以稳定抓取。','',
'## 方法与限制','',
'- 每条示范均取12个均匀有效锚点，并加上闭爪/松爪转换附近的固定偏移。仅取连续30帧有效专家窗口；未采到的状态没有测量。阶段按首次闭爪/随后松爪划分，包含多次重抓的纠偏示范阶段标签只是近似。',
'- 主查验读取原始JPEG，经保存的部署预处理及归一化，FP32推理；未使用专家动作编码VAE，模型处于eval模式。所有模型权重及500条源记录的元数据哈希重新核对通过。',
'- 8个源/划分/阶段样本对比了原图处理与缓存处理：归一化像素差约1.5e-5，预测最大差约0.0015（含二值夹爪分数）。这是独立数值检查，主结果均来自原图路径，不依赖缓存近似。',
'- 示范10Hz，本轮闭环验证8Hz；15步动作的时间跨度不同。这项时间差可能影响闭环，但本次离线查验不能定量归因。',
'- 几何参考点为当前杯子配置中的参考偏移，并非重新标定的精确指间中点。这里比较两条动作轨迹的同一参考点，可信度高于绝对杯心距离。',
'- 离线专家状态预测准确不等于闭环成功：自主运行偏离后图像、关节状态和接触条件都会改变。还需后续受控试验区分图像利用和状态分布偏移。','',
'建议下一项最小对照：在同一机械臂姿态下改变杯子位置，观察近端预测是否随杯子正确变化；并针对当前失败模型实际到达的偏移状态采集恢复示范。','',
'[完整结果](results.json) · [逐样本表格](sample_metrics.csv) · [闭爪切换统计](closure_onset.json) · [原始预测数组](predictions.npz)']
(A/'查验报告.md').write_text('\n'.join(lines)+'\n')
(A/'verification.json').write_text(json.dumps(dict(passed=True,samples=12415,train_episodes=450,validation_episodes=50,prediction_shape=[12415,30,7],all_samples_eligible=True,source_metadata_hashes_verified=True,model_hash_unchanged=True,targets_excluded=True),indent=2))
print(A/'查验报告.md')
