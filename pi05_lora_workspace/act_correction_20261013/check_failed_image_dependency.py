"""Fixed RGB/arm-state counterfactual gripper inference, no robot commands."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch,av
from torch.utils.data import default_collate
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
RUN=Path(__file__).resolve().parent;OUT=RUN/'failed_image_dependency_20261014';OUT.mkdir(exist_ok=True)
CP='/home/lp531/smolvla_training/act_correction_chunk16_20261013/checkpoints/050000/pretrained_model'
policy=ACTPolicy.from_pretrained(CP,local_files_only=True,strict=True).eval();pre,post=make_pre_post_processors(policy.config,pretrained_path=CP)
angles=[-.9,-.895,-.88,-.85,-.8,-.75,-.7,0.]
results=[]
def check(label,batch,metadata):
 conditions=['original','gray_wrist','gray_global','gray_global_right','gray_all']
 b={'observation.state':batch['observation.state'].repeat(5,1).clone()}
 for key,image in batch.items():
  if not key.startswith('observation.images.'):continue
  v=image.repeat(5,1,1,1);v[conditions.index('gray_'+key.split('.')[-1])]=.5;v[4]=.5;b[key]=v
 policy.reset()
 with torch.inference_mode():p=post(policy.predict_action_chunk(pre(b))).cpu().numpy()
 entries=[]
 for i,name in enumerate(conditions):
  g=p[i,:,-1];entries.append(dict(condition=name,first_prediction=float(g[0]),prediction=g.tolist(),first_close_step_06=next((j for j,v in enumerate(g) if v>=.6),None)))
 results.append(dict(label=label,metadata=metadata,fixed_state=batch['observation.state'][0].tolist(),entries=entries))
 (OUT/'predictions.json').write_text(json.dumps(results,indent=2));print('checked',label,flush=True)
pose=json.loads((RUN/'failed_pose_check_20261014/report.json').read_text())
for seed in [1126247665,437216237]:
 rr=next(x for x in pose['episodes'] if x['seed']==seed);folder=Path(rr['result']).parent;trace=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()];sync=[json.loads(x) for x in (folder/'input_sync.jsonl').read_text().splitlines()]
 starts=[i for i,x in enumerate(trace) if x['policy_action_index']==0 and x['execution_substep']==0 and not x.get('policy_action_paused')]
 assert len(starts)==len(sync),(len(starts),len(sync))
 # One video snapshot before each prediction, plus one after each completed action.
 frames=[];completed=0;si=0
 for i,x in enumerate(trace):
  if si<len(starts) and i==starts[si]:frames.append(si+completed);si+=1
  if x['execution_substep']==5 and not x.get('policy_action_paused'):completed+=1
 target_time=rr['closest_to_preclose']['time_s'];rank=sorted(range(len(starts)),key=lambda j:abs(trace[starts[j]]['simulation_elapsed_s']-target_time))[:3]
 for cam,file in [('wrist','wrist.mp4'),('global','global_third_person.mp4'),('global_right','global_right.mp4')]:
  wanted={frames[j]:j for j in rank};decoded={}
  with av.open(str(folder/'videos'/file)) as c:
   for fi,frame in enumerate(c.decode(video=0)):
    if fi in wanted:decoded[wanted[fi]]=torch.from_numpy(frame.to_ndarray(format='rgb24').copy()).permute(2,0,1).float().unsqueeze(0)/255
    if len(decoded)==len(wanted):break
  assert len(decoded)==len(wanted)
  if cam=='wrist':batches={j:{} for j in rank}
  for j in rank:batches[j]['observation.images.'+cam]=decoded[j]
 for j in sorted(rank):
  batches[j]['observation.state']=torch.tensor([sync[j]['input_state']],dtype=torch.float32)
  check('eval_'+str(seed)+'_query'+str(j),batches[j],dict(type='failed_eval',seed=seed,query=j,video_frame=frames[j],time_s=trace[starts[j]]['simulation_elapsed_s'],recorded_first_prediction=trace[starts[j]]['policy_action'][-1],images_exact=False,image_note='Lossy MP4 snapshot immediately before query; synchronized policy image tensor not saved, so replay is approximate.'))
print('FINISHED',len(results),flush=True)
