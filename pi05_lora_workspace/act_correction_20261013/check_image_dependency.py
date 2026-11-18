"""ACT RGB interventions with all seven state dimensions held fixed."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
RUN=Path(__file__).resolve().parent;OUT=RUN/'image_dependency_20261014';OUT.mkdir(exist_ok=True)
CP='/home/lp531/smolvla_training/act_correction_chunk16_20261013/checkpoints/050000/pretrained_model'
policy=ACTPolicy.from_pretrained(CP,local_files_only=True,strict=True).eval();pre,post=make_pre_post_processors(policy.config,pretrained_path=CP)
split=json.loads((RUN/'split_manifest.json').read_text());cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='lp531/z1_bottle_with_fragments_20261012',root='/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012',episodes=split['validation_episodes'],video_backend='pyav'),policy=policy.config);cfg.dataset.image_transforms.enable=False;val=make_dataset(cfg)
with np.load(RUN/'sampling_arrays.npz') as a:eps=a['episode'].copy()
conditions=['original','gray_wrist','gray_global','gray_global_right','gray_all','before_1s_images','after_1s_images']
results=[]
for ep in split['validation_episodes']:
 rec=next(x for x in split['source_records'] if x['episode_index']==ep);start=int(np.flatnonzero(eps==ep)[0]);close=rec['close_frame']
 samples={d:default_collate([val[val.absolute_to_relative_idx[start+close+d]]]) for d in [-10,0,10]}
 state=samples[0]['observation.state'].repeat(len(conditions),1).clone();batch={'observation.state':state}
 for cam in val.meta.camera_keys:
  image=samples[0][cam].float()/255;v=image.repeat(len(conditions),1,1,1)
  short=cam.split('.')[-1];v[conditions.index('gray_'+short)]=.5;v[conditions.index('gray_all')]=.5
  v[conditions.index('before_1s_images')]=samples[-10][cam].float()[0]/255
  v[conditions.index('after_1s_images')]=samples[10][cam].float()[0]/255
  batch[cam]=v
 assert torch.equal(state,state[0:1].repeat(len(conditions),1))
 policy.reset()
 with torch.inference_mode():p=post(policy.predict_action_chunk(pre(batch))).cpu().numpy()
 entries=[]
 for i,name in enumerate(conditions):
  g=p[i,:,-1];entries.append(dict(condition=name,gripper_prediction=g.tolist(),first_prediction=float(g[0]),first_close_step_06=next((j for j,v in enumerate(g) if v>=.6),None),closed_within8=bool((g[:8]>=.6).any()),arm_prediction=p[i,:,:6].tolist()))
 results.append(dict(episode=ep,source=rec['source'],frame=close,fixed_state=state[0].tolist(),state_sha256=hashlib.sha256(state[0].numpy().tobytes()).hexdigest(),entries=entries))
 (OUT/'predictions.json').write_text(json.dumps(results,indent=2));print('checked',ep,flush=True)
summary=[]
for name in conditions:
 e=[next(x for x in r['entries'] if x['condition']==name) for r in results];base=[r['entries'][0] for r in results]
 p=np.array([x['first_prediction'] for x in e]);b=np.array([x['first_prediction'] for x in base]);changes=np.array([np.mean(np.abs(np.array(x['gripper_prediction'])-np.array(y['gripper_prediction']))) for x,y in zip(e,base)])
 summary.append(dict(condition=name,observations=len(e),closed_first_06=int((p>=.6).sum()),closed_within8=sum(x['closed_within8'] for x in e),first_prediction_mean=float(p.mean()),mean_abs_first_change=float(np.abs(p-b).mean()),max_abs_first_change=float(np.abs(p-b).max()),first_decision_flips=int(((p>=.6)!=(b>=.6)).sum()),mean_abs_chunk_change=float(changes.mean())))
(OUT/'summary.json').write_text(json.dumps(dict(checkpoint=CP,observations=len(results),predictions=len(results)*len(conditions),summary=summary,robot_commands_sent=False),indent=2))
with (OUT/'predictions.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['episode','condition','horizon','gripper_prediction'])
 for r in results:
  for e in r['entries']:
   for h,p in enumerate(e['gripper_prediction']):w.writerow([r['episode'],e['condition'],h,p])
print(json.dumps(summary,indent=2),flush=True)
