"""Held-out PI05 close-boundary inference with deployment preprocessing."""
import json,csv,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from peft import PeftModel
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import load_base_policy,reset_policy
RUN=Path(__file__).resolve().parent;OUT=RUN/'validation_gripper_20261014';OUT.mkdir(exist_ok=True)
CP='/home/lp531/smolvla_training/pi05_fine_grip_recovery_base_20261012/checkpoints/120000/pretrained_model'
config=PreTrainedConfig.from_pretrained(CP,local_files_only=True);config.device='cuda'
adapter=json.loads((Path(CP)/'adapter_config.json').read_text());policy=load_base_policy(PI05Policy,adapter['base_model_name_or_path'],config);policy=PeftModel.from_pretrained(policy,CP,is_trainable=False,local_files_only=True).eval()
pre,post=make_pre_post_processors(config,pretrained_path=CP,preprocessor_overrides={'device_processor':{'device':config.device}})
records=[x for x in json.loads((RUN/'phase_manifest.json').read_text()) if x['split']=='validation'];eps=[r['episode_index'] for r in records]
cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='lp531/z1_bottle_with_fragments_20261012',root='/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012',episodes=eps,video_backend='pyav'),policy=config);cfg.dataset.image_transforms.enable=False;val=make_dataset(cfg)
rows=[];repeat=[]
for rec in records:
 ep=rec['episode_index'];start=int(val.meta.episodes[ep]['dataset_from_index']);close=rec['close_frame']
 for offset in [-10,-5,-2,-1,0,1,2,5,10,15,20]:
  frame=close+offset
  if frame<0 or frame+14>=rec['frame_count']:continue
  sample=default_collate([val[val.absolute_to_relative_idx[start+frame]]]);batch={'observation.state':sample['observation.state'],'task':['Physically grasp the bottle, lift it clear, and place it in the box.']}
  for cam in val.meta.camera_keys:batch[cam]=sample[cam].float()/255
  for seed in ([20261014,20261015,20261016] if offset==0 else [20261014]):
   reset_policy(policy,seed)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=post(policy.predict_action_chunk(pre(batch))).detach().float().cpu().numpy()[0,:,-1]
   entry=dict(episode=ep,source=rec['source'],frame=frame,close_frame=close,offset=offset,noise_seed=seed,state_gripper=float(sample['observation.state'][0,-1]),truth=sample['action'][0,:,-1].tolist(),prediction=pred.tolist(),first_prediction=float(pred[0]),first_truth=float(sample['action'][0,0,-1]),first_close_step_06=next((i for i,v in enumerate(pred) if v>=.6),None),closed_within15=bool((pred>=.6).any()))
   (rows if seed==20261014 else repeat).append(entry)
  (OUT/'predictions.json').write_text(json.dumps(dict(primary=rows,onset_repeats=repeat),indent=2))
 print('completed episode',ep,flush=True)
summary={}
for label,subset in [('before',[r for r in rows if r['offset']<0]),('at_onset',[r for r in rows if r['offset']==0]),('after_onset',[r for r in rows if r['offset']>0]),('at_and_after',[r for r in rows if r['offset']>=0]),('after_1s',[r for r in rows if r['offset']>=10])]:
 p=np.array([r['first_prediction'] for r in subset]);summary[label]=dict(observations=len(subset),first_closed_05=int((p>=.5).sum()),first_closed_06=int((p>=.6).sum()),chunk_contains_close=sum(r['closed_within15'] for r in subset),min=float(p.min()),max=float(p.max()),mean=float(p.mean()),median=float(np.median(p)),near_05=int(((p>=.45)&(p<=.55)).sum()))
report=dict(checkpoint=CP,validation_episodes=eps,source_counts={s:sum(r['source']==s for r in records) for s in ['original','old_correction','new_fragment']},primary_observations=len(rows),extra_onset_predictions=len(repeat),summary=summary,noise_seed=20261014,robot_commands_sent=False)
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
with (OUT/'predictions.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['episode','source','offset','seed','horizon','truth','prediction'])
 for r in rows+repeat:
  for i,(t,p) in enumerate(zip(r['truth'],r['prediction'])):w.writerow([r['episode'],r['source'],r['offset'],r['noise_seed'],i,t,p])
print(json.dumps(report,indent=2),flush=True)
