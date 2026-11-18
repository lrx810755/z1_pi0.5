from pathlib import Path
import json,collections,csv
import numpy as np
r=Path(__file__).parent
out=[]
for group in ('pd','position_iterations_double','velocity_iterations_increase'):
 d=r/group
 if not (d/'telemetry.jsonl').exists():continue
 buckets=collections.defaultdict(list)
 for line in (d/'telemetry.jsonl').read_text().splitlines():
  x=json.loads(line)
  if x['phase']=='hold':buckets[(x['variant'],x['pose'])].append(x)
 for (variant,pose),rows in buckets.items():
  tail=[x for x in rows if x['elapsed']>=13]
  if not tail:continue
  q=np.array([x['q'] for x in tail]);target=np.array([x['target'] for x in tail]);qd=np.array([x['qd'] for x in tail]);err=q-target
  out.append(dict(variant=variant,pose=pose,n=len(tail),tail_start=tail[0]['elapsed'],tail_end=tail[-1]['elapsed'],
                  joint3_mean_error_rad=float(err[:,2].mean()),joint3_abs_error_rad=float(np.abs(err[:,2]).mean()),
                  joint3_p95_abs_error_rad=float(np.quantile(np.abs(err[:,2]),.95)),joint3_position_peak_to_peak_rad=float(np.ptp(q[:,2])),
                  joint3_reported_velocity_rms_rad_s=float(np.sqrt((qd[:,2]**2).mean())),
                  arm_p95_max_abs_error_rad=float(np.quantile(np.abs(err[:,:6]).max(axis=1),.95)),
                  fraction_all_arm_below_012=float((np.abs(err[:,:6]).max(axis=1)<=.012).mean())))
(r/'results.json').write_text(json.dumps(out,indent=2))
if out:
 with (r/'results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
for x in out:print(x['variant'],x['pose'],round(x['joint3_abs_error_rad'],7),round(x['arm_p95_max_abs_error_rad'],7))
