from pathlib import Path
import json,collections,csv
R=Path(__file__).resolve().parent
ms=json.loads((R/'metrics.json').read_text())
groups={8:{'全程未执行闭爪':[4,6,7,10,13,14,16,23,27,29,30],'闭爪偏位，未形成可靠夹持':[1,3,5,11,12,17,19,21,26,28],'接触后夹持不稳或低位停滞':[2,15,20,24,25],'抬起后未完成搬运':[],'搬运途中闭爪滑落':[],'接近阶段夹指碰桌':[9,18]},15:{'全程未执行闭爪':[6,14],'闭爪偏位，未形成可靠夹持':[1,3,7,9,16,20,21,24,27],'接触后夹持不稳或低位停滞':[4,5,12,15,25],'抬起后未完成搬运':[10,11,28],'搬运途中闭爪滑落':[8,19],'接近阶段夹指碰桌':[18]}}
for steps,classes in groups.items():
 ids=[n for group in classes.values() for n in group]
 assert len(ids)==len(set(ids))
 assert set(ids)=={m['episode'] for m in ms if m['steps']==steps and not m['success']}
 for m in ms:
  if m['steps']==steps:m['review_category']='成功' if m['success'] else next(k for k,ids in classes.items() if m['episode'] in ids)
(R/'reviewed_metrics.json').write_text(json.dumps(ms,ensure_ascii=False,indent=2))
keys=['steps','episode','seed','success','category','review_category','close_events','open_events','first_close_s','min_envelope_error_m','both_seconds','max_root_z','final_root','chunks_predict_close','chunks_close_beyond_execution','path']
with (R/'逐次结果.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
 for m in sorted(ms,key=lambda x:(x['steps'],x['episode'])):w.writerow({k:m[k] for k in keys})
lines=['# 苹果ACT 8步/15步评估失败复核','', '核对全部60次trace.jsonl、prediction_chunks.jsonl和result.json，并抽查14个案例的三视角关键帧（42路视频）；关键帧以记录动作序列与MP4帧数一一核对后定位。视频关键帧在本目录frames_*.jpg。','', '## 复核分类（每次失败只分配一个主类）','','|具体表现|8步|15步|','|---|---:|---:|']
for k in groups[8]:lines.append(f'|{k}|{len(groups[8][k])}|{len(groups[15][k])}|')
lines+=['','8步2/30成功；15步8/30成功。原始评估结果不修改；本表是动作表现的人工归类，并非对评估器重打成功标签。','', '## 证据与边界','', '1. 8步第27次共51次预测都将首个>=0.6的闭爪输出放在0基索引9～15（第10～16步），实际只执行前8步，整轮没有发出闭爪目标。第13/16/29次也存在只在被截掉的后半段预测闭爪的情况。其余7次全程不闭爪的失败，所有预测chunk均未达到0.6。该统计排除另外2次碰桌提前终止。','', '2. 闭爪偏位是几何/接触证据，不意味着已经证明视觉网络完全不看图。例如8步第12次首闭爪时相对夹爪局部Y=0.057611m，原苹果示范夹持包络是±0.007m；8步第1次夹爪接触杯子，未接触苹果。前者可证偏位，不能单凭它区分视觉定位、姿态偏差、轨迹控制等训练根因。','', '3. 低位停滞：15步第4/5/12/15/25次有双指苹果接触，但根节点最高分别约10.4/7.4/12.4/7.2/7.2mm。后三阶段长时间小调整，未继续形成明确的高位搬运。8步第20次升到47.7mm后也未继续搬运；8步第2次是3次闭爪、2次开爪，接触短暂中断；8步第25次仅约1.4s双指接触后丢失。因此表中使用“夹持不稳或低位停滞”，不把所有情况笼统称为没闭爪。','', '4. 抬升到搬运的转换停滞：15步第10/11/28次苹果根节点最高约92.0/122.0/96.1mm，60秒结束时仍没入盒，均无开爪目标。末端与模型目标都只做缓慢小幅调整，末段目标/实际关节误差P95约0.012/0.020/0.019rad，不能归因于“完全没执行到模型要求的位置”。目前证据支持阶段推进停滞，不足以确定其训练根因。','', '5. 原评估器把15步第8/19次归为placement_failed；实际是途中滑落。第8次根节点24.32s达到136.6mm，24.72s已落至20mm以下，30.15s才首次开爪；第19次38.42s达到148.2mm，39.85s已落至20mm以下，45.25s才开爪。两次落下时execution_target[-1]仍为0（闭爪），视频显示苹果在蓝盒外落桌，手臂随后空爪继续前往盒子。不能把它们解释为到盒后不松爪或提前开爪。摩擦、夹持深度、姿态变化各自贡献未做控制对照，暂不归因。','', '6. 反复张闭是叠加现象：8步第2/21次；15步第7/8/16/20/21/27次。15步第8次反复开闭发生在苹果已经滑落之后，不能当作该次最初掉落的原因。','', '7. 干扰物接触：8步第1/5/9/16/29次和15步第9/16/24次均存在累计>0.2s的夹指接触非目标物。15步第16次夹指接触杯子约48.25s，8步第5次接触鼠标约7.27s。仅凭接触不推断模型把杯子当成苹果，但可确认偏位/干扰物阻挡实际出现。','', '8. 系统性推理或通信异常：这60次error_detail均为空，完整输出视频与日志。并不表示每个物理判据都已被人工重新定标。','', '## 逐次编号','']
for steps,classes in groups.items():
 lines.append(f'### 执行{steps}步')
 for name,ids in classes.items():lines.append(f'- {name}：{", ".join(map(str,ids)) or "无"}')
 lines.append('')
(R/'具体失败原因.md').write_text('\n'.join(lines))
print('classified',sum(len(ids) for cls in groups.values() for ids in cls.values()),'failures')
