import json,sys,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from geometry import audit,height
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
from z1_isaac_bridge.smolvla_rollout_core import JOINT_LIMITS
ROOT=Path(__file__).resolve().parent;torch.set_num_threads(1)
CHECKPOINT=Path('/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints/080000/pretrained_model')
os.environ['Z1_PI05_INFERENCE_SEED']='123';policy=PI05ProcessPolicy(CHECKPOINT)
try:
 for label in json.loads((ROOT/'selected.json').read_text()):
  source=Path(label['raw_root']);rows=[json.loads(x) for x in (source/'frames.jsonl').read_text().splitlines()];c=label['close_frame']
  points=[('home',0),('motion_start',label['first_motion_frame']),('approach25',c//4),('approach50',c//2),('approach75',3*c//4),('preclose',c-1)]
  for stage,i in points:
   row=rows[i];batch={'observation.state':torch.tensor([row['observation.state']],dtype=torch.float32),'task':[row['task']]}
   for camera in ('wrist','global','global_right'):
    key='observation.images.'+camera;array=np.array(Image.open(source/row[key]).convert('RGB'));batch[key]=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255
   ref=np.array([x['action'] for x in rows[i:i+15]]);ref[:,-1]=[float(c<=j<label['open_frame']) for j in range(i,i+15)]
   policy.reset();started=time.monotonic();pred=policy.predict_action_chunk(batch)[0].numpy();elapsed=time.monotonic()-started
   mapped=pred.copy();mapped[:,-1]=np.where(mapped[:,-1]>=.5,0,-.9);clipped=np.clip(mapped,JOINT_LIMITS[:,0],JOINT_LIMITS[:,1]);state=np.array(row['observation.state'])
   poserr=[float(np.linalg.norm(audit.fk(a[:6])[0]-audit.fk(b[:6])[0])*1000) for a,b in zip(pred,ref)]
   item=dict(episode=label['raw_episode'],stage=stage,frame=i,state=state.tolist(),reference=ref.tolist(),prediction=pred.tolist(),position_error_mm=poserr,first_action_max_joint_error_rad=float(np.max(abs(pred[0,:6]-ref[0,:6]))),joint_mae_rad=float(np.mean(abs(pred[:,:6]-ref[:,:6]))),raw_limit_change_rad=np.max(abs(mapped[:,:6]-clipped[:,:6]),axis=1).tolist(),raw_height_m=[height(a) for a in mapped],clipped_height_m=[height(a) for a in clipped],inference_s=elapsed)
   with (ROOT/'offline_predictions.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
   print(label['raw_episode'],stage,'first_error_mm',round(poserr[0],2),'limit_rad',round(max(item['raw_limit_change_rad']),3),flush=True)
finally:policy.close()
(ROOT/'offline_complete.json').write_text(json.dumps(dict(observations=36,checkpoint=str(CHECKPOINT),scope='training fitting diagnostic; not held-out validation',seed=123)))
