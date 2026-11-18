from pathlib import Path
import json,math
r=Path(__file__).parent
assert json.loads((r/'all_completed.json').read_text())['complete']
names=['baseline','P_half','P_double','D_half','D_double','effort_double','position_iterations_double','velocity_iterations_increase','baseline_repeat']
base=json.loads((r/'baseline/runtime_parameters.json').read_text())
checks=[]
for name in names:
 d=r/name;assert json.loads((d/'exit.json').read_text())['code']==0
 p=json.loads((d/'runtime_parameters.json').read_text())
 assert p['targets']==base['targets'] and p['limits']==base['limits']
 for j in range(7):
  assert p['gains'][0][j]==base['gains'][0][j]*(.5 if name=='P_half' and j==2 else 2 if name=='P_double' and j==2 else 1)
  assert p['gains'][1][j]==base['gains'][1][j]*(.5 if name=='D_half' and j==2 else 2 if name=='D_double' and j==2 else 1)
  assert p['max_efforts'][j]==base['max_efforts'][j]*(2 if name=='effort_double' and j==2 else 1)
 assert p['solver_position_iterations']==([64] if name=='position_iterations_double' else [32])
 assert p['solver_velocity_iterations']==([8] if name=='velocity_iterations_increase' else [1])
 a=[json.loads(l) for l in (d/'telemetry.jsonl').read_text().splitlines()]
 assert a[-1]['t']>=26
 assert all(x['gains']==p['gains'] for x in a)
 assert all(math.isfinite(v) for x in a for v in x['q']+x['qd'])
 checks.append({'group':name,'samples':len(a),'last_simulation_time':a[-1]['t'],'parameter_check':True})
out=json.loads((r/'results.json').read_text());assert len(out)==18
for pose in ('hold_preclose','hold_transport'):
 b=next(x for x in out if x['variant']=='baseline' and x['pose']==pose)
 b2=next(x for x in out if x['variant']=='baseline_repeat' and x['pose']==pose)
 assert abs(b['joint3_mean_abs_error_rad']-b2['joint3_mean_abs_error_rad'])<1e-5
(r/'verification.json').write_text(json.dumps({'groups':checks,'completed_pose_comparisons':18,'baseline_reproducible':True},indent=2))
print('9组参数核验通过；18个姿态结果完整；重复基线一致')
