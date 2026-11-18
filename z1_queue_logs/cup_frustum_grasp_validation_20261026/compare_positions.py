import json
from pathlib import Path
R=Path(__file__).parent
out=[]
for p in sorted((R/'cases').glob('*/candidate.json')):
 d=p.parent
 if not (d/'configured.json').exists() or not (d/'result.json').exists():continue
 config=json.loads((d/'configured.json').read_text());candidate=json.loads(p.read_text())['candidate'];reference=config['physical']['calibrated_grasp_reference']['object_in_gripper_local_m'];desired=candidate.get('desired_object_in_stator_m')
 events=[json.loads(s) for s in (d/'assistant_events.jsonl').read_text().splitlines()];close=[e['sim_time'] for e in events if e.get('phase')=='GUIDED_CLOSE'];probes=[json.loads(s) for s in (d/'scene_events.jsonl').read_text().splitlines()];probes=[e for e in probes if e.get('phase')=='PROBED' and e.get('local_position_m')]
 selected=[e for e in probes if close and e['sim_time']<=close[0]];probe=selected[-1] if selected else (probes[0] if probes else {})
 out.append(dict(case=d.name,nominal_reference_local_m=reference,final_candidate_local_m=desired,preclose_probe_local_m=probe.get('local_position_m'),preclose_probe_time_s=probe.get('sim_time'),grasp_command_time_s=close[0] if close else None,probe_selection='last PROBED before GUIDED_CLOSE' if selected else 'first PROBED; no earlier close command',passed=json.loads((d/'result.json').read_text())['passed']))
(R/'闭爪前标称_候选_实测.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
lines=['# 闭爪前位置对照','', '原候选搜索可额外叠加局部位置偏移；因此修改配置的基准参考，不等于机械臂每次停在该点。下面是多朝向局部范围实验的实际记录，单位均为毫米。实测取闭爪命令前最后一次Isaac PROBED状态。','', '| 条目 | 标称X,Z | 最终候选X,Z | 实测X,Z | 流程通过 |','|---|---|---|---|---|']
for e in out:
 if not e['case'].startswith('range_flexible'):continue
 def xz(v):return ', '.join(f'{v[i]*1000:.3f}' for i in [0,2]) if v else '无'
 lines.append(f"| {e['case']} | {xz(e['nominal_reference_local_m'])} | {xz(e['final_candidate_local_m'])} | {xz(e['preclose_probe_local_m'])} | {'是' if e['passed'] else '否'} |")
(R/'闭爪前位置对照.md').write_text('\n'.join(lines)+'\n');print('Position comparisons',len(out))
