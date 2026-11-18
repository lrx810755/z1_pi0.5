from pathlib import Path
import json,sys,os
import numpy as np
import torch
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
OUT=Path(__file__).resolve().parent
root=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012')
records=json.loads(Path('/home/lp531/pi05_lora_workspace/fine_grip_recovery_base_20261012/phase_manifest.json').read_text())
table=pq.read_table(root/'data',columns=['index','episode_index','action']).sort_by([('index','ascending')])
episodes=np.array(table['episode_index']);actions=np.array(table['action'].to_pylist())
ds=LeRobotDataset('lp531/z1_bottle_with_fragments_20261012',root=root,video_backend='pyav')
samples=[]
for ep in [0,40,80,120,160,180,198,203]:
 r=records[ep];start=int(np.flatnonzero(episodes==ep)[0]);end=int(np.flatnonzero(episodes==ep)[-1])+1
 for offset in [-5,0,5]:
  frame=r['close_frame']+offset;idx=start+frame;item=ds[idx]
  batch={'observation.state':item['observation.state'].unsqueeze(0).float(),'task':[item['task']]}
  for key in ds.meta.camera_keys:
   im=item[key].float()
   if im.max()>1:im=im/255
   batch[key]=im.unsqueeze(0)
  ref=actions[idx:min(idx+15,end)]
  samples.append(dict(episode=ep,split=r['split'],offset=offset,frame=frame,batch=batch,reference=ref.tolist()))
print('SAMPLES',len(samples),flush=True)
os.environ['Z1_PI05_INFERENCE_SEED']='20261013'
models=[('old80k','pi05_gentle_correction_base_20261010','080000'),('new80k','pi05_fine_grip_recovery_base_20261012','080000'),('new120k','pi05_fine_grip_recovery_base_20261012','120000')]
with (OUT/'offline_predictions.jsonl').open('w') as f:
 for label,run,step in models:
  print('LOADING',label,flush=True)
  policy=PI05ProcessPolicy(Path('/home/lp531/smolvla_training')/run/'checkpoints'/step/'pretrained_model')
  try:
   for sample in samples:
    policy.reset();pred=policy.predict_action_chunk(sample['batch'])[0].numpy()
    row={k:v for k,v in sample.items() if k!='batch'}
    row.update(model=label,state=sample['batch']['observation.state'][0].tolist(),predicted=pred.tolist())
    f.write(json.dumps(row)+'\n');f.flush()
   print('FINISHED',label,flush=True)
  finally:policy.close()
print('COMPLETE',flush=True)
