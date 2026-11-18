"""Offline read-only: infer the deployed ACT on actual demonstration states/images."""
import os,sys,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
TRAIN=Path('/home/lp531/pi05_lora_workspace/act_cup_minus90_full_cached_20261023');sys.path.insert(0,str(TRAIN))
from cache_data import CupDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.processor import ImageCropResizeProcessorStep
R=Path(__file__).resolve().parent
checkpoint='/home/lp531/smolvla_training/act_cup_minus90_full_cached_chunk16_20261023/checkpoints/050000/pretrained_model'
torch.set_num_threads(4)
policy=ACTPolicy.from_pretrained(checkpoint,local_files_only=True);policy.eval()
pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=checkpoint)
# The immutable cache is already the exact crop/resize result (pixel error 0).
pre.steps=[s for s in pre.steps if not isinstance(s,ImageCropResizeProcessorStep)]
manifest=json.loads((TRAIN/'split_manifest.json').read_text());ds=CupDataset(list(range(len(manifest['source_records']))))
items=[]
for rec in manifest['source_records']:
 for name,frame in [('before_close_05',rec['close_frame']-5),('before_close_01',rec['close_frame']-1),('after_close_03',rec['close_frame']+3),('before_release_05',rec['open_frame']-5),('after_release_03',rec['open_frame']+3)]:
  if frame>=0 and frame+15<rec['frames']:items.append(dict(episode=rec['episode_index'],source=rec['source'],split=rec['split'],event=name,frame=frame,absolute=rec['dataset_from_index']+frame))
results=[]
with torch.inference_mode():
 for start in range(0,len(items),8):
  group=items[start:start+8];batch=default_collate([ds[x['absolute']] for x in group]);truth=batch['action'].clone()
  processed=pre(batch);processed.pop('action',None);processed.pop('action_is_pad',None)
  prediction=post(policy.predict_action_chunk(processed)).detach().cpu()
  for j,x in enumerate(group):
   gt=truth[j];pred=prediction[j];arm=(pred[:15,:6]-gt[:15,:6]).abs()
   onset_close=np.flatnonzero(pred[:15,-1].numpy()>=.6);onset_open=np.flatnonzero(pred[:15,-1].numpy()<=.4)
   results.append(dict(**x,first_gripper=float(pred[0,-1]),first_arm_mae_rad=float(arm[0].mean()),chunk15_arm_mae_rad=float(arm.mean()),predicted_close_first_step=int(onset_close[0]) if len(onset_close) else None,predicted_open_first_step=int(onset_open[0]) if len(onset_open) else None,gripper_predictions=pred[:,-1].tolist(),truth_gripper=gt[:,-1].tolist()))
  if start%160==0:print('Inferred',min(start+8,len(items)),'/',len(items),flush=True)
summary={}
for split in ['train','validation']:
 summary[split]={}
 for event in ['before_close_05','before_close_01','after_close_03','before_release_05','after_release_03']:
  group=[x for x in results if x['split']==split and x['event']==event]
  if not group:continue
  summary[split][event]=dict(n=len(group),gripper_first_median=float(np.median([x['first_gripper'] for x in group])),close_first_fraction=float(np.mean([x['first_gripper']>=.6 for x in group])),open_first_fraction=float(np.mean([x['first_gripper']<=.4 for x in group])),first_arm_mae_rad_median=float(np.median([x['first_arm_mae_rad'] for x in group])),chunk15_arm_mae_rad_median=float(np.median([x['chunk15_arm_mae_rad'] for x in group])))
(R/'demo_state_inference.json').write_text(json.dumps(dict(checkpoint=checkpoint,uses_real_demonstration_observations=True,inference_mode=True,vae_latent_zero=True,uses_training_targets_for_prediction=False,summary=summary,results=results),indent=2))
print(json.dumps(summary,indent=2))
