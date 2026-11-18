import json,csv,statistics,math
from pathlib import Path
r=Path(__file__).parent
names=['baseline','P_half','P_double','D_half','D_double','effort_double','position_iterations_double','velocity_iterations_increase','baseline_repeat']
results=[]
for name in names:
 d=r/name;p=d/'telemetry.jsonl'
 if not p.exists() or not (d/'runtime_parameters.json').exists():continue
 a=[json.loads(l) for l in p.read_text().splitlines()];pars=json.loads((d/'runtime_parameters.json').read_text())
 for phase in ('hold_preclose','hold_transport'):
  rows=[x for x in a if x['phase']==phase]
  if not rows or rows[-1]['t'] < (13 if phase=='hold_preclose' else 25):continue
  end=rows[-1]['t'];rows=[x for x in rows if x['t']>=end-3]
  errs=[[abs(q-t) for q,t in zip(x['q'][:6],x['target'][:6])] for x in rows]
  peaks=sorted(max(e) for e in errs);e3=[e[2] for e in errs]
  results.append(dict(variant=name,pose=phase,kp=pars['gains'][0][2],kd=pars['gains'][1][2],max_effort=pars['max_efforts'][2],
   solver_position=pars['solver_position_iterations'][0],solver_velocity=pars['solver_velocity_iterations'][0],
   joint3_target_rad=rows[-1]['target'][2],joint3_mean_abs_error_rad=statistics.mean(e3),joint3_median_abs_error_rad=statistics.median(e3),
   joint3_q_peak_to_peak_rad=max(x['q'][2] for x in rows)-min(x['q'][2] for x in rows),
   joint3_reported_velocity_rms_rad_s=math.sqrt(statistics.mean(x['qd'][2]**2 for x in rows)),
   all_arm_max_error_p95_rad=peaks[min(len(peaks)-1,int(len(peaks)*.95))],
   all_arm_in_012_fraction=sum(x<=.012 for x in peaks)/len(peaks), samples=len(rows)))
(r/'results.json').write_text(json.dumps(results,indent=2))
if results:
 with (r/'results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
for x in results:print(x['variant'],x['pose'],round(x['joint3_mean_abs_error_rad'],7),'arm',round(x['all_arm_max_error_p95_rad'],7))
