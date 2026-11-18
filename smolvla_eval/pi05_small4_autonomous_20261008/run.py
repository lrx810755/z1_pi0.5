"""Paired offline fitting check, then 12 fully autonomous training-scene trials."""
import json,sys,os,time,threading
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import rclpy
from rclpy.executors import MultiThreadedExecutor
sys.path.insert(0,'/home/lp531/smolvla_eval/pi05_action_contract_audit_20261007')
from geometry import audit,height
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout,parse_args
from z1_isaac_bridge.smolvla_rollout_core import JOINT_LIMITS,execution_window
OUT=Path(__file__).resolve().parent
LABELS=json.loads((OUT/'selected.json').read_text())
SMALL=Path('/home/lp531/smolvla_training/pi05_bottle_small4_chunk15_attn_ffn_20261007/checkpoints/008000/pretrained_model')
LARGE=Path('/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints/080000/pretrained_model')
torch.set_num_threads(1)

def offline(policy,name):
 for label in LABELS:
  source=Path(label['raw_root']);rows=[json.loads(x) for x in (source/'frames.jsonl').read_text().splitlines()];c=label['close_frame'];o=label['open_frame']
  points=[('home',0),('motion_start',label['first_motion_frame']),('approach25',c//4),('approach50',c//2),('approach75',3*c//4),('preclose',c-1),('close',c),('lift',c+20),('prerelease',o-1),('release',o+5)]
  for stage,i in points:
   assert i+15<=len(rows)
   row=rows[i];batch={'observation.state':torch.tensor([row['observation.state']],dtype=torch.float32),'task':[row['task']]}
   for cam in ('wrist','global','global_right'):
    key='observation.images.'+cam;array=np.array(Image.open(source/row[key]).convert('RGB'));batch[key]=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float()/255
   ref=np.array([x['action'] for x in rows[i:i+15]]);ref[:,-1]=[float(c<=j<o) for j in range(i,i+15)]
   policy.reset();started=time.monotonic();pred=policy.predict_action_chunk(batch)[0].numpy();elapsed=time.monotonic()-started
   assert pred.shape==(15,7) and np.isfinite(pred).all()
   mapped=pred.copy();mapped[:,-1]=np.where(mapped[:,-1]>=.5,0,-.9);clipped=np.clip(mapped,JOINT_LIMITS[:,0],JOINT_LIMITS[:,1])
   poserr=[float(np.linalg.norm(audit.fk(a[:6])[0]-audit.fk(b[:6])[0])*1000) for a,b in zip(pred,ref)]
   firstclose=next((j+1 for j,a in enumerate(pred) if a[-1]>=.5),None)
   item=dict(model=name,episode=label['raw_episode'],stage=stage,frame=i,state=row['observation.state'],reference=ref.tolist(),prediction=pred.tolist(),position_error_mm=poserr,joint_mae_rad=float(np.mean(abs(pred[:,:6]-ref[:,:6]))),raw_limit_change_rad=np.max(abs(mapped[:,:6]-clipped[:,:6]),axis=1).tolist(),raw_height_m=[height(a) for a in mapped],clipped_height_m=[height(a) for a in clipped],first_close_frame=firstclose,gripper_correct=((pred[:,-1]>=.5)==(ref[:,-1]>=.5)).tolist(),inference_s=elapsed)
   with (OUT/'offline_predictions.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
   print('OFFLINE',name,label['raw_episode'],stage,'first_mm',round(poserr[0],2),'close',firstclose,flush=True)

class Autonomous(SmolVLASimRollout):
 def _randomize_scene(self,target,fixed_seed=None):
  original=self._randomization_command_pub;grid=self.recorded_scene['target_grid']
  class Publisher:
   def publish(self,message):
    payload=json.loads(message.data);payload['target_grid']=grid;message.data=json.dumps(payload);original.publish(message)
  self._randomization_command_pub=Publisher()
  try:return super()._randomize_scene(target,fixed_seed)
  finally:self._randomization_command_pub=original
 def _prepare_episode(self,target,fixed_seed=None):
  seed=super()._prepare_episode(target,fixed_seed)
  before=self.recorded_scene['object_positions_m'];after=self._randomization_status['object_positions_m']
  self.layout_errors={name:float(np.linalg.norm(np.array(v)-after[name])) for name,v in before.items()}
  if max(self.layout_errors.values())>.005:raise RuntimeError('scene_restore_position_mismatch: '+json.dumps(self.layout_errors))
  return seed
 def _predict(self,batch):
  started=time.perf_counter()
  with torch.inference_mode():full=self.policy.predict_action_chunk(batch).detach().float().cpu().numpy()
  elapsed=time.perf_counter()-started;root=self._input_sync_path.parent;index=self._input_sync_index-1
  archive=root/'policy_inputs';archive.mkdir(exist_ok=True)
  arrays={}
  for key,value in batch.items():
   if key.startswith('observation.images.'):arrays[key]=(value.detach().cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
   elif key=='observation.state':arrays[key]=value.detach().cpu().numpy()
  np.savez(archive/f'{index:04d}.npz',**arrays)
  with (root/'predictions.jsonl').open('a') as f:f.write(json.dumps(dict(prediction=index,input_state=arrays['observation.state'].tolist(),task=batch['task'],full_actions=full[0].tolist(),execute_steps=self.args.execute_steps,inference_wall_s=elapsed,recording_wall_s=time.perf_counter()-started-elapsed))+'\n')
  return execution_window(full,self.args.execute_steps)

def main():
 if (OUT/'started.json').exists():raise RuntimeError('refusing to overwrite existing experiment')
 plan=dict(scope='training scene memorization check, not held-out validation',small_checkpoint=str(SMALL),large_checkpoint=str(LARGE),offline_observations_per_model=40,execute_steps=5,chunk=15,action_hz=10,execution_hz=60,timeout_sim_s=60,repeats=3,inference_seed=123,frozen_bottle=False,contact_wait=0,reference_arm_assistance=False,forced_gripper=False,scenes=[dict(episode=x['raw_episode'],seed=x['seed'],grid=x['grid']) for x in LABELS])
 (OUT/'started.json').write_text(json.dumps(plan,indent=2))
 print('LOAD_LARGE_BASELINE',flush=True);baseline=PI05ProcessPolicy(LARGE)
 try:offline(baseline,'large160_080000')
 finally:baseline.close()
 args=parse_args(['--policy-type','pi05','--checkpoint',str(SMALL),'--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--reject-blank-cameras','--execute-steps','5','--rate','10','--execution-rate','60','--prelift-contact-stable-s','0','--no-freeze-target-until-close','--binary-gripper','--gripper-action-encoding','binary','--timeout','60','--output',str(OUT)])
 args.training_manifest=None
 rclpy.init();node=None;executor=None;results=[]
 try:
  node=Autonomous(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
  offline(node.policy,'small4_008000')
  (OUT/'offline_complete.json').write_text(json.dumps(dict(observations=80)))
  for repeat in range(1,4):
   for label in LABELS:
    source=Path(label['raw_root']);node.recorded_scene=json.loads((source/'episode_start.json').read_text())['scene_randomization']
    trial=OUT/f"repeat{repeat}"/label['raw_episode'];trial.mkdir(parents=True,exist_ok=False)
    print('BEGIN_AUTONOMOUS',repeat,label['raw_episode'],flush=True)
    result=node.run_episode('beverage_bottle',len(results)+1,trial,fixed_seed=label['seed'])
    result.update(repetition=repeat,source_episode=label['raw_episode'],layout_errors_m=node.layout_errors,autonomous=True)
    results.append(result);(OUT/'results.json').write_text(json.dumps(results,indent=2))
    print('RESULT_AUTONOMOUS',repeat,label['raw_episode'],result['success'],result['stop_reason'],result['error_detail'],result['physical_predicates'],flush=True)
    if result['stop_reason'] in {'runtime_error','stale_stream','input_sync_timeout','stale_grasp_status','blank_camera','invalid_policy_action'}:raise RuntimeError('infrastructure failure: '+result['error_detail'])
  (OUT/'completed.json').write_text(json.dumps(dict(trials=len(results),successes=sum(x['success'] for x in results)),indent=2))
 finally:
  if node:node._set_active(False)
  if executor:executor.shutdown();thread.join(timeout=2)
  if node:node.policy.close();node.destroy_node()
  if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
