"""Offline held-out demonstration gripper diagnostic; no ROS commands."""
import json,csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
RUN=Path(__file__).resolve().parent
OUT=RUN/'validation_gripper_20261014';OUT.mkdir(exist_ok=True)
CP='/home/lp531/smolvla_training/act_correction_chunk16_20261013/checkpoints/050000/pretrained_model'
split=json.loads((RUN/'split_manifest.json').read_text())
policy=ACTPolicy.from_pretrained(CP,local_files_only=True,strict=True).eval()
pre,post=make_pre_post_processors(policy.config,pretrained_path=CP)
cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='lp531/z1_bottle_with_fragments_20261012',root='/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012',episodes=split['validation_episodes'],video_backend='pyav'),policy=policy.config)
cfg.dataset.image_transforms.enable=False
val=make_dataset(cfg)
with np.load(RUN/'sampling_arrays.npz') as a: episodes=a['episode'].copy();frames=a['frame'].copy()
rows=[]
for ep in split['validation_episodes']:
 rec=next(r for r in split['source_records'] if r['episode_index']==ep)
 close=rec['close_frame'];ids=np.flatnonzero(episodes==ep);start=int(ids[0]);n=len(ids)
 offsets=[-10,-5,-2,-1,0,1,2,5,10,15,20]
 anchors=[close+d for d in offsets if 0<=close+d and close+d+15<n]
 samples=[val[val.absolute_to_relative_idx[start+f]] for f in anchors]
 sample=default_collate(samples)
 batch={'observation.state':sample['observation.state']}
 for cam in val.meta.camera_keys:batch[cam]=sample[cam].float()/255
 policy.reset()
 with torch.inference_mode():pred=post(policy.predict_action_chunk(pre(batch))).cpu().numpy()
 truth=sample['action'].numpy()
 for k,f in enumerate(anchors):
  p=pred[k,:,-1];t=truth[k,:,-1]
  rows.append(dict(episode=ep,source=rec['source'],frame=f,close_frame=close,offset=f-close,
   actual_state_gripper=float(sample['observation.state'][k,-1]),truth=t.tolist(),prediction=p.tolist(),
   first_prediction=float(p[0]),first_truth=float(t[0]),predicted_close_05=bool(p[0]>=.5),predicted_close_06=bool(p[0]>=.6)))
 (OUT/'predictions.json').write_text(json.dumps(rows,indent=2))
 print('completed episode',ep,'windows',len(anchors),flush=True)
summary={}
for name,subset in [('before',[r for r in rows if r['offset']<0]),('at_and_after',[r for r in rows if r['offset']>=0]),('after_1s',[r for r in rows if r['offset']>=10])]:
 p=np.array([r['first_prediction'] for r in subset]);t=np.array([r['first_truth'] for r in subset]);summary[name]=dict(windows=len(subset),prediction_min=float(p.min()),prediction_max=float(p.max()),prediction_mean=float(p.mean()),prediction_median=float(np.median(p)),truth_closed=int((t==1).sum()),predicted_closed_05=int((p>=.5).sum()),predicted_closed_06=int((p>=.6).sum()),near_05_045_055=int(((p>=.45)&(p<=.55)).sum()),in_hysteresis_band=int(((p>.4)&(p<.6)).sum()))
per=[]
for ep in split['validation_episodes']:
 rr=[r for r in rows if r['episode']==ep and r['offset']>=0];per.append(dict(episode=ep,source=rr[0]['source'],first_step_max=max(r['first_prediction'] for r in rr),first_step_min=min(r['first_prediction'] for r in rr),closed_05=sum(r['predicted_close_05'] for r in rr),windows=len(rr)))
report=dict(checkpoint=CP,validation_episodes=split['validation_episodes'],fps=10,offset_frames=[-10,-5,-2,-1,0,1,2,5,10,15,20],windows=len(rows),summary=summary,episodes=per,robot_commands_sent=False)
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
with (OUT/'predictions.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['episode','source','offset','frame','horizon','truth','prediction'])
 for r in rows:
  for h,(t,p) in enumerate(zip(r['truth'],r['prediction'])):w.writerow([r['episode'],r['source'],r['offset'],r['frame'],h,t,p])
print(json.dumps(report,indent=2),flush=True)
