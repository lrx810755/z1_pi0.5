import json
from pathlib import Path
from collections import Counter
p=Path(__file__).resolve().parent
rs=json.loads((p/'results.json').read_text()); ss=json.loads((p/'summary.json').read_text()); ms=json.loads((p/'analysis_metrics.json').read_text())
assert json.loads((p/'completed.json').read_text())['model_trials']==15
assert all(sum(x['stage']==g for x in rs)==5 for g in ['home','midway','preclose'])
lines=['# PI0.5 固定场景分段接管实验','', '日期：2026-10-04。第一步实验已完成：参考回放预检后，Home、接近中段、正确闭爪前姿态各接管 5 次，共 15 次真实模型推理。', '', '## 方法与有效性','', '- 权重：原 LoRA 的 108000 checkpoint；本次没有训练或切换权重。', '- 场景 seed=342470491，grid=[3,0]；每条完整复位。模型随机种子固定 20261004，三组交错执行。这是固定场景重复性诊断，不是独立随机场景成功率估计。', '- 模型使用实时双相机 RGB 和关节状态，预测 50 点，每次执行前 5 点；动作 10 Hz，线性插值执行 60 Hz。', '- 接管位置来自可成功的参考轨迹第 0、50、105 帧。引导全过程夹爪保持张开，接管时清空模型状态及抓取证据；没有人为提前闭爪。', '- 保留生产触桌/跟踪保护，闭爪前固定瓶子且开启碰撞；闭爪后沿用现有解冻及接触稳定门控。', '- 参考预检第 1 次因双侧接触未持续稳定而失败，第 2 次完整成功，随后进入模型实验。参考执行仍存在物理重复性问题，不能称为稳定 100% 成功。', '', '## 结果','', '| 接管起点 | 次数 | 执行闭爪 | 物理抬起 | 完整成功 | 停止原因 |','|---|---:|---:|---:|---:|---|']
for g,name in [('home','Home'),('midway','接近中段'),('preclose','正确闭爪前')]:
 a=[x for x in ss if x['stage']==g]; b=[x for x in ms if x['stage']==g]
 lines.append(f"| {name} | {len(a)} | {sum(x['first_close_sim_s'] is not None for x in a)} | {sum(x['lifted'] for x in b)} | {sum(x['success'] for x in a)} | {dict(Counter(x['stop_reason'] for x in a))} |")
lines += ['', '## 完整 chunk 与实际执行的区别','', '以下动作序号从 1 开始；日志索引从 0 开始。','', '| 闭爪前组 | 首个 chunk 的首次闭爪动作 | 所有 chunk 中前 5 点含闭爪的次数 | 接管后首次离开抓取范围（仿真秒） |','|---|---:|---:|---:|']
for x in [x for x in ms if x['stage']=='preclose']:
 s=next(s for s in ss if s['stage']=='preclose' and s['repeat']==x['repeat'])
 lines.append(f"| 第 {x['repeat']} 次 | {x['first_close_index']+1 if x['first_close_index'] is not None else '无'} | {s['chunks_close_in_first5']} | {x['first_outside_envelope_s']:.3f} |")
model=[x for x in ss if x['stage']!='reference']; mets=[x for x in ms if x['stage']!='reference']
lines += ['', f"模型共记录 {sum(x['prediction_chunks'] for x in model)} 个完整 chunk；其中 {sum(x['chunks_close_only_later'] for x in model)} 个在第 6–50 点预测了闭爪，{sum(x['chunks_close_in_first5'] for x in model)} 个在前 5 点预测闭爪。", '', '首轮闭爪前组的闭爪索引随重规划为：'+str(next(x for x in ms if x['stage']=='preclose')['close_offsets'])+'（0 起始）。', '', '这支持“滚动执行时闭爪不断后移”的诊断：模型能预测未来闭爪，但当前执行窗口始终取到张开阶段，同时手臂继续改变位置。它并不能证明只增大执行窗口就能完成任务；未来动作也可能不准确，必须另做对照。', '', '接管几何已校验，但几何包围区不等于真实稳定夹持。腕部画面可见瓶子，仍有明显近距离视角与遮挡；本实验没有做相机消融，不能断言视觉问题已排除。', '', '## 执行侧观察与局限','', f"- 15 次接管的最大手臂姿态误差为 {max(x['handoff_arm_error_rad'] for x in mets):.6f} rad；记录的终点命令保持时间为 {min(x['endpoint_hold_s'] for x in mets):.3f}–{max(x['endpoint_hold_s'] for x in mets):.3f} 仿真秒。", f"- 模型阶段实测夹爪角度范围 {min(x['actual_jaw_range'][0] for x in mets):.4f} 至 {max(x['actual_jaw_range'][1] for x in mets):.4f} rad，部分明显偏离约 -0.9 rad 的原始张开预测。限步器以实测状态计算命令，因此日志中的最终命令与原始模型输出不可混同。偏差原因仍需单独查驱动/碰撞/限步响应。", '- 同一个推理随机种子下，动态接触与异步时序仍会带来轨迹差异；5 次重复用于判断现象是否复现。', '- 没有通过降低闭爪阈值、关闭碰撞或放宽触桌保护制造成功。', '', '## 下一步建议','', '优先做不重新训练的执行窗口对照：仅从已验证的闭爪前姿态开始，比较 execute_steps=5 与 15（必要时再加 20），保持 50 点预测、10 Hz 动作、60 Hz 执行及全部保护一致。检查是否真正闭爪、是否仍在抓取范围、是否形成双侧接触并抬起。不要直接在全部验证入口永久加长 chunk。', '', '同时单独检查夹爪保持 -0.9 rad 时的实测响应，定位实测角度偏离及限步器影响。若较长窗口能闭爪但仍碰撞/掉落，再分辨预测轨迹与夹持物理；若仍不断返回接近阶段，再针对阶段转换和数据分布选择训练改动。此次结果不足以支持直接增加数据条数或训练步数。', '', '## 文件','', '- `started.json`：固定参数、权重位置、参考动作 SHA256。', '- `results.json` / `summary.json` / `analysis_metrics.json`：完整结果及统计。', '- `home/`、`midway/`、`preclose/`：各 5 次；包含交接图像、完整预测、引导轨迹、执行轨迹及双路视频。', '- `reference/`：两次参考预检，包含失败和成功证据。', '- `evaluate.py` / `run.sh`：实验程序；`completed.json` 为完成标记。', '', '本次只新增独立实验程序和报告，没有修改生产 SH 的默认行为。']
(p/'分段接管实验报告.md').write_text('\n'.join(lines)+'\n')
print(p/'分段接管实验报告.md')
