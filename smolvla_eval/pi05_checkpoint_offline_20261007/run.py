"""Paired offline checkpoint experiment; never publishes robot commands."""
from pathlib import Path
import json,sys,time,struct,importlib.util,os
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
from z1_isaac_bridge.smolvla_rollout_core import JOINT_LIMITS,binary_gripper_target,safe_joint_target
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit','/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
torch.set_num_threads(1)
meshes={}
for name in ('Stator','Mover'):
 data=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_description/meshes/collision/z1_Gripper'+name+'.STL').read_bytes()
 dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')]);n=struct.unpack_from('<I',data,80)[0]
 vertices=np.unique(np.frombuffer(data,dtype=dtype,count=n,offset=84)['vertices'].reshape(-1,3),axis=0)
 # Convex hull vertices preserve the minimum of a linear projection and speed up height queries.
 from scipy.spatial import ConvexHull
 meshes[name]=vertices[ConvexHull(vertices).vertices]
def height(q):
 pos,r=audit.fk(q[:6]);v=meshes['Stator']@r.T+pos
 rm=Rotation.from_rotvec(np.array([0,q[-1],0])).as_matrix();vm=(meshes['Mover']@rm.T+np.array([.049,0,0]))@r.T+pos
 return float(min(v[:,2].min(),vm[:,2].min()))
def executed(actions,current):
 current=np.array(current,dtype=float);out=[]
 for action in actions:
  goal=binary_gripper_target(action,encoding='binary');start=current.copy();endpoint=safe_joint_target(goal,start,.1)
  for sub in range(6):
   command=start+(endpoint-start)*(sub+1)/6
   command[-1]=goal[-1]
   current=safe_joint_target(command,current,.1);out.append(current.copy())
 return np.array(out)
def mapped(actions):
 a=np.array(actions,dtype=float).copy();a[:,-1]=np.where(a[:,-1]>=.5,0.,-.9);return a
labels=json.loads(Path('/home/lp531/pi05_lora_workspace/binary_gripper_20261007/episode_labels.json').read_text())
# Farthest-point selection spreads the chosen observations over the demonstrated bottle positions.
positions=np.array([json.loads((Path(e['raw_root'])/'episode_start.json').read_text())['scene_randomization']['object_positions_m']['beverage_bottle'][:2] for e in labels])
selected=[int(np.argmin(np.linalg.norm(positions-np.median(positions,axis=0),axis=1)))]
while len(selected)<24:
 distances=np.min(np.linalg.norm(positions[:,None,:]-positions[selected][None,:,:],axis=2),axis=1)
 distances[selected]=-1;selected.append(int(np.argmax(distances)))
samples=[]
for index in selected:
 e=labels[index];root=Path(e['raw_root']);rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
 for offset in (30,15,1):
  i=e['close_frame']-offset
  assert i>=0 and i+15<=len(rows)
  reference=np.array([r['action'] for r in rows[i:i+15]],dtype=float)
  reference[:,-1]=[float(e['close_frame']<=j<e['open_frame']) for j in range(i,i+15)]
  reference_original=np.array([r['action'] for r in rows[i:i+15]],dtype=float)
  batch={'observation.state':torch.tensor([rows[i]['observation.state']],dtype=torch.float32),'task':[rows[i]['task']]}
  for camera in ('wrist','global','global_right'):
   key='observation.images.'+camera;array=np.array(Image.open(root/rows[i][key]).convert('RGB'))
   batch[key]=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255
  samples.append(dict(episode=e['raw_episode'],frame=i,offset=offset,position=positions[index].tolist(),batch=batch,reference=reference,reference_original=reference_original,state=rows[i]['observation.state']))
manifest={'scope':'training-set fitting check; all 160 episodes were used in training; no held-out success estimate','selection':'24 episodes, farthest-point sampling of bottle xy; close onset minus 30,15,1 frames','episodes':24,'observations':len(samples),'noise_seeds':[123],'checkpoints':['008000','040000','080000'],'height_method':'URDF gripper collision meshes, base z=0/table config top_z=0; geometric proxy, not Isaac contact or physics replay','risk_thresholds_m':[0,.004],'execution_reconstruction':'10Hz actions /60Hz substeps, .1rad limit, assumes perfect tracking of last command; not physical execution','samples':[dict(episode=s['episode'],frame=s['frame'],offset=s['offset'],position=s['position']) for s in samples]}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert height([0]*6+[-.9])==height([0]*6+[-.9])
os.environ['Z1_PI05_INFERENCE_SEED']='123'
base=Path('/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints')
for checkpoint in manifest['checkpoints']:
 print('LOADING',checkpoint,flush=True)
 policy=PI05ProcessPolicy(base/checkpoint/'pretrained_model')
 try:
  for n,s in enumerate(samples):
   ref=s['reference'];original=s['reference_original'];state=s['state'];refmap=mapped(ref)
   ref_heights=np.array([height(q) for q in refmap]);original_heights=np.array([height(q) for q in original]);refposes=[audit.fk(q[:6]) for q in ref]
   for seed in manifest['noise_seeds']:
    policy.reset()
    started=time.perf_counter();pred=policy.predict_action_chunk(s['batch'])[0].numpy();elapsed=time.perf_counter()-started
    assert pred.shape==(15,7) and np.isfinite(pred).all()
    raw=mapped(pred);clipped=np.clip(raw,JOINT_LIMITS[:,0],JOINT_LIMITS[:,1]);commands=executed(pred,state)
    raw_h=np.array([height(q) for q in raw]);clipped_h=np.array([height(q) for q in clipped]);command_h=np.array([height(q) for q in commands])
    poses=[audit.fk(q[:6]) for q in raw];poserr=np.array([np.linalg.norm(p[0]-r[0]) for p,r in zip(poses,refposes)])
    roterr=np.array([Rotation.from_matrix(p[1]@r[1].T).magnitude() for p,r in zip(poses,refposes)])
    first=lambda a:next((j+1 for j,v in enumerate(a) if v>=.5),None)
    item=dict(checkpoint=checkpoint,episode=s['episode'],frame=s['frame'],offset=s['offset'],seed=seed,inference_wall_s=elapsed,state=state,predicted=pred.tolist(),reference=ref.tolist(),joint_mae_rad=float(np.mean(abs(pred[:,:6]-ref[:,:6]))),joint_rmse_rad=float(np.sqrt(np.mean((pred[:,:6]-ref[:,:6])**2))),position_error_m=poserr.tolist(),rotation_error_rad=roterr.tolist(),gripper_correct=((pred[:,-1]>=.5)==(ref[:,-1]>=.5)).tolist(),predicted_first_close=first(pred[:,-1]),reference_first_close=first(ref[:,-1]),joint_exceeds=np.any((pred[:,:6]<JOINT_LIMITS[:6,0]-1e-5)|(pred[:,:6]>JOINT_LIMITS[:6,1]+1e-5),axis=0).tolist(),raw_heights=raw_h.tolist(),clipped_heights=clipped_h.tolist(),command_heights=command_h.tolist(),reference_heights=ref_heights.tolist(),reference_original_heights=original_heights.tolist(),reference_command_heights=[height(q) for q in executed(ref,state)])
    with (OUT/'predictions.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
   print('PROGRESS',checkpoint,n+1,'/',len(samples),flush=True)
 finally:policy.close()
print('COMPLETE',flush=True)
