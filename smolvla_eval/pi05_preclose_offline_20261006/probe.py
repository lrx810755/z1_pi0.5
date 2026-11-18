"""Offline timing probe at physically stationary, open-gripper demonstration states."""
from pathlib import Path
import importlib.util,json,os,sys,time
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from PIL import Image
OUT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
spec=importlib.util.spec_from_file_location('audit_fk','/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006/audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
manifest=json.loads(Path('/home/lp531/pi05_lora_workspace/three_camera_20261006/split_manifest.json').read_text())
samples=[];excluded=[]
for e in manifest['episodes']:
 p=Path(e['source']);rows=[json.loads(l) for l in (p/'frames.jsonl').read_text().splitlines()]
 onset=next(i for i,r in enumerate(rows) if r['action'][-1]>=-.89)
 closed=next(i for i,r in enumerate(rows) if r['action'][-1]>=-.75)
 i=onset-1
 if i<5:excluded.append({'episode':p.name,'reason':'short_preclose_history'});continue
 pose,rot=audit.fk(rows[closed]['observation.state'][:6]);history=[audit.fk(r['observation.state'][:6]) for r in rows[i-5:i+1]]
 position_error=max(float(np.linalg.norm(pos-pose)) for pos,_ in history)
 rotation_error=max(float(np.rad2deg(Rotation.from_matrix(rr@rot.T).magnitude())) for _,rr in history)
 if position_error>.002 or rotation_error>1 or rows[i]['observation.state'][-1]>-.88:
  excluded.append({'episode':p.name,'reason':'not_stationary_open_at_close_pose','position_error_m':position_error,'rotation_error_deg':rotation_error});continue
 samples.append((e,p,rows,i,position_error,rotation_error))
(OUT/'selection.json').write_text(json.dumps({'selected':len(samples),'excluded':excluded,'criteria':'Last sample before closure onset; open gripper <= -0.88 rad; preceding 0.5 s within 2 mm and 1 degree of actual pose when demonstration first reaches -0.75 rad'},indent=2)+'\n')
print('Selected',len(samples),'excluded',len(excluded),flush=True)
os.environ['Z1_PI05_INFERENCE_SEED']='123'
policy=PI05ProcessPolicy(Path('/home/lp531/smolvla_training/pi05_bottle_three_camera_20261006/best'))
results=[];started=time.monotonic()
try:
 for n,(e,p,rows,i,pe,re) in enumerate(samples):
  r=rows[i];batch={'observation.state':torch.tensor([r['observation.state']],dtype=torch.float32),'task':[e['task']]}
  for name in ['wrist','global','global_right']:
   k='observation.images.'+name;im=np.array(Image.open(p/r[k]).convert('RGB'));batch[k]=torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float()/255
  seeds=[123]
  for seed in seeds:
   policy.reset()
   a=policy.predict_action_chunk(batch)[0].numpy()
   first=lambda values,threshold:next((j+1 for j,v in enumerate(values) if v>=threshold),None)
   item={'episode':p.name,'split':e['split'],'seed':seed,'frame':i,'input_gripper':r['observation.state'][-1],
    'position_error_m':pe,'rotation_error_deg':re,
    'pred_first_close_frame':first(a[:,-1],-.75),'pred_first_closing_onset_frame':first(a[:,-1],-.85),
    'reference_first_close_frame':first([x['action'][-1] for x in rows[i:i+50]],-.75),
    'pred_gripper':a[:,-1].tolist(),'reference_gripper':[x['action'][-1] for x in rows[i:i+50]]}
   results.append(item)
   with (OUT/'predictions.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
  print('PROBED',n+1,'/',len(samples),p.name,results[-1]['pred_first_close_frame'],flush=True)
finally:policy.close()
summary={'checkpoint':str(Path('/home/lp531/smolvla_training/pi05_bottle_three_camera_20261006/best').resolve()),'selected_episodes':len(samples),'excluded_episodes':len(excluded),'predictions':len(results),'elapsed_s':time.monotonic()-started,'scope':'Offline saved synchronized three-camera input; no robot commands','splits':{}}
for split in ['train','validation']:
 data=[r for r in results if r['split']==split]
 summary['splits'][split]={'samples':len(data),'within5':sum(r['pred_first_close_frame'] is not None and r['pred_first_close_frame']<=5 for r in data),'within10':sum(r['pred_first_close_frame'] is not None and r['pred_first_close_frame']<=10 for r in data),'within15':sum(r['pred_first_close_frame'] is not None and r['pred_first_close_frame']<=15 for r in data),'no_close_in50':sum(r['pred_first_close_frame'] is None for r in data),'first_close_frame_quantiles':np.quantile([r['pred_first_close_frame'] for r in data if r['pred_first_close_frame'] is not None],[0,.25,.5,.75,1]).tolist() if any(r['pred_first_close_frame'] is not None for r in data) else [],'reference_within5':sum(r['reference_first_close_frame'] is not None and r['reference_first_close_frame']<=5 for r in data)}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
