from pathlib import Path
from collections import defaultdict
import json,csv
import numpy as np
OUT=Path(__file__).resolve().parent
records=[json.loads(x) for x in (OUT/'predictions.jsonl').read_text().splitlines()]
groups=defaultdict(list)
for r in records:groups[r['checkpoint']].append(r)
summary=[]
for ck,rs in sorted(groups.items()):
 assert len(rs)==72 and len({(r['episode'],r['frame'],r['seed']) for r in rs})==72
 item={'checkpoint':ck,'observations':len(rs),'joint_mae_deg':float(np.rad2deg(np.mean([r['joint_mae_rad'] for r in rs]))),'joint_rmse_deg':float(np.rad2deg(np.sqrt(np.mean([r['joint_rmse_rad']**2 for r in rs])))),'position_mae_mm':float(np.mean([r['position_error_m'] for r in rs])*1000),'rotation_mae_deg':float(np.rad2deg(np.mean([r['rotation_error_rad'] for r in rs]))),'gripper_accuracy':float(np.mean([r['gripper_correct'] for r in rs])),'joint_exceeding_chunks':np.sum([r['joint_exceeds'] for r in rs],axis=0).tolist(),'infer_wall_median_ms':float(np.median([r['inference_wall_s'] for r in rs][1:])*1000),'infer_wall_p95_ms':float(np.percentile([r['inference_wall_s'] for r in rs][1:],95)*1000),'windows':{},'phases':{}}
 for horizon in (5,10,15):
  raw=np.array([r['raw_heights'][:horizon] for r in rs]);clip=np.array([r['clipped_heights'][:horizon] for r in rs]);cmd=np.array([r['command_heights'][:horizon*6] for r in rs]);ref=np.array([r['reference_heights'][:horizon] for r in rs]);orig=np.array([r['reference_original_heights'][:horizon] for r in rs]);refcmd=np.array([r['reference_command_heights'][:horizon*6] for r in rs])
  win={}
  for threshold in (0.,.004):
   key='below_table' if threshold==0 else 'under_4mm'
   safe=(ref.min(axis=1)>=threshold)
   win[key]={'raw_chunks':int(np.sum(raw.min(axis=1)<threshold)),'clipped_chunks':int(np.sum(clip.min(axis=1)<threshold)),'command_chunks':int(np.sum(cmd.min(axis=1)<threshold)),'reference_binary_chunks':int(np.sum(ref.min(axis=1)<threshold)),'reference_original_chunks':int(np.sum(orig.min(axis=1)<threshold)),'reference_command_chunks':int(np.sum(refcmd.min(axis=1)<threshold)),'reference_safe_observations':int(safe.sum()),'raw_bad_on_reference_safe':int(np.sum((raw.min(axis=1)<threshold)&safe)),'clipped_bad_on_reference_safe':int(np.sum((clip.min(axis=1)<threshold)&safe))}
  win['raw_min_height_mm']=float(raw.min()*1000);win['clipped_min_height_mm']=float(clip.min()*1000);win['command_min_height_mm']=float(cmd.min()*1000);win['reference_min_height_mm']=float(ref.min()*1000)
  item['windows'][str(horizon)]=win
 for offset in (30,15,1):
  phase=[r for r in rs if r['offset']==offset]
  item['phases'][str(offset)]={'n':len(phase),'joint_mae_deg':float(np.rad2deg(np.mean([r['joint_mae_rad'] for r in phase]))),'raw_below_table_chunks':sum(min(r['raw_heights'])<0 for r in phase),'raw_under4mm_chunks':sum(min(r['raw_heights'])<.004 for r in phase),'clipped_under4mm_chunks':sum(min(r['clipped_heights'])<.004 for r in phase),'close_in5':sum(r['predicted_first_close'] is not None and r['predicted_first_close']<=5 for r in phase),'close_in10':sum(r['predicted_first_close'] is not None and r['predicted_first_close']<=10 for r in phase),'close_in15':sum(r['predicted_first_close'] is not None for r in phase),'reference_close_in15':sum(r['reference_first_close'] is not None for r in phase)}
 summary.append(item)
result={'scope':'training-set offline paired fitting check; no physics replay or closed-loop success claim','summary':summary}
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (OUT/'per_observation.csv').open('w') as f:
 fields=['checkpoint','episode','frame','offset','joint_mae_deg','position_mae_mm','raw_min_mm','clipped_min_mm','command_min_mm','reference_min_mm','first_close','reference_first_close']
 w=csv.DictWriter(f,fields);w.writeheader()
 for r in records:w.writerow(dict(checkpoint=r['checkpoint'],episode=r['episode'],frame=r['frame'],offset=r['offset'],joint_mae_deg=np.rad2deg(r['joint_mae_rad']),position_mae_mm=np.mean(r['position_error_m'])*1000,raw_min_mm=min(r['raw_heights'])*1000,clipped_min_mm=min(r['clipped_heights'])*1000,command_min_mm=min(r['command_heights'])*1000,reference_min_mm=min(r['reference_heights'])*1000,first_close=r['predicted_first_close'],reference_first_close=r['reference_first_close']))
print(json.dumps(result,indent=2))
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,3,figsize=(14,4))
 for ck,rs in sorted(groups.items()):
  axs[0].plot(range(1,16),np.mean([r['position_error_m'] for r in rs],axis=0)*1000,label=ck)
  axs[1].plot(range(1,16),np.mean([r['gripper_correct'] for r in rs],axis=0),label=ck)
  axs[2].plot(range(1,16),np.mean(np.array([r['clipped_heights'] for r in rs])<.004,axis=0),label=ck)
 for ax,title,ylabel in zip(axs,['Position error vs demonstration','Binary gripper agreement','Clipped targets under 4 mm'],['Mean error (mm)','Accuracy','Fraction of observations']):
  ax.set(title=title,xlabel='Predicted action index',ylabel=ylabel);ax.grid(alpha=.3);ax.legend()
 fig.tight_layout();fig.savefig(OUT/'checkpoint_comparison.png',dpi=160)
except ImportError:pass
