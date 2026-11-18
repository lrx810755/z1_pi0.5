"""Bounded replay experiments; preserve source frames and arm protection."""
import argparse,json,sys,threading,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
sys.path.insert(0,'/home/lp531/smolvla_eval/pi05_four_step_validation_20261003')
from replay import Replay,DATA
from z1_isaac_bridge.smolvla_sim_rollout import parse_args,JOINT_NAMES
from z1_isaac_bridge.trajectory_bridge import JOINT_LIMITS
from z1_isaac_bridge.smolvla_rollout_core import GraspPhaseTracker
BASE=Path(__file__).resolve().parent
ROOT=BASE
class DebugReplay(Replay):
 def _scene_grasp_callback(self,msg):
  super()._scene_grasp_callback(msg)
  try:payload=json.loads(msg.data)
  except ValueError:return
  if payload.get('phase')=='CONTACT_STATE':
   with self._lock:self.contact=payload
 def freeze(self,keep):
  with self._lock:self._scene_grasp_reply={}
  msg=String();msg.data=json.dumps({'command':'freeze','target':'beverage_bottle','keep_collisions':keep});self._scene_grasp_pub.publish(msg)
  end=time.monotonic()+5
  while time.monotonic()<end:
   with self._lock:reply=dict(self._scene_grasp_reply)
   if reply.get('phase')=='FROZEN' and reply.get('target_collisions_enabled') is keep:return
   time.sleep(.01)
  raise RuntimeError('freeze mode not acknowledged')
 def wait_until(self,stamp):
  end=time.monotonic()+5
  while time.monotonic()<end:
   with self._lock:now=self._joint_sim_stamp
   if now>=stamp-1e-8:return
   time.sleep(.002)
  raise RuntimeError('simulation clock stopped')
 def trial(self,index,label,keep=True,hz=10,release='threshold',duration=None):
  folder=getattr(self,'data_root',DATA)/f'episode_{index:06d}';start=json.loads((folder/'episode_start.json').read_text());rows=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
  self.recorded_scene=start['scene_randomization'];self._publish_target('beverage_bottle');self._randomize_scene('beverage_bottle',self.recorded_scene['seed']);self._set_active(True);self.freeze(keep)
  commands=np.array([r['action'] for r in rows]);states=np.array([r['observation.state'] for r in rows]);limits=np.array([JOINT_LIMITS[n] for n in JOINT_NAMES]);onset=next(i for i,a in enumerate(commands) if a[-1]>commands[0,-1]+.002);threshold=next(i for i,a in enumerate(commands) if a[-1]>=-.75)
  source_rate=float(start.get('fps',10));end_time=min(duration if duration is not None else len(rows)/source_rate,len(rows)/source_rate);trace=[];tracker=GraspPhaseTracker();frozen=True;fault_since=None;reason='trajectory_complete';previous=None;time_shift=0.;settled=False
  with self._lock:origin=self._joint_sim_stamp;initial=[self._joints[n] for n in JOINT_NAMES]
  try:
   for tick in range(int(end_time*hz)):
    t=tick/hz;self.wait_until(origin+t+time_shift)
    images,actual,status,evidence=self._snapshot()
    if previous is not None:
     error=float(np.max(np.abs(previous[:6]-actual[:6])))
     with self._lock:sim=self._joint_sim_stamp
     if error>.08:fault_since=sim if fault_since is None else fault_since
     else:fault_since=None
     if error>.20 or (fault_since is not None and sim-fault_since>.20):reason='joint_tracking_fault';break
    if status.get('left_finger_touching_table') or status.get('right_finger_touching_table'):reason='finger_table_contact';break
    if getattr(self,'settle_before_lift',False) and not settled and t>=12.1:
     stable_since=None;settle_origin=self._joint_sim_stamp
     while self._joint_sim_stamp-settle_origin<1.0:
      images,actual,status,evidence=self._snapshot()
      sim=self._joint_sim_stamp
      if status.get('bilateral_contact'):
       stable_since=sim if stable_since is None else stable_since
      else:stable_since=None
      if stable_since is not None and sim-stable_since>=.5:break
      if previous is not None:self._publish_command(previous)
      self.wait_until(sim+1/60)
     if stable_since is None or self._joint_sim_stamp-stable_since<.5:
      reason='prelift_contact_not_stable';break
     time_shift+=self._joint_sim_stamp-settle_origin;settled=True
    release_time=start['recorded_unfreeze_s'] if release=='recorded' else (onset if release=='onset' else threshold)/source_rate
    if frozen and t+1e-6>=release_time:
     self._set_target_frozen('beverage_bottle',False);frozen=False
    x=t*source_rate;lo=min(int(x),len(rows)-1);hi=min(lo+1,len(rows)-1);ratio=x-lo
    requested=commands[lo] if hz==source_rate else commands[lo]*(1-ratio)+commands[hi]*ratio
    command=np.clip(requested,limits[:,0],limits[:,1]);self._publish_grasp_phase('beverage_bottle',tracker.update(command[-1],actual[-1],status));self._publish_command(command);previous=command
    with self._lock:contact=dict(getattr(self,'contact',{}));sim=self._joint_sim_stamp
    trace.append({'scheduled_s':t,'simulation_elapsed_s':sim-origin,'source_index':lo,'source_phase':rows[lo]['phase'],'requested':requested.tolist(),'command':command.tolist(),'joint_state':actual.tolist(),'source_state':(states[lo]*(1-ratio)+states[hi]*ratio).tolist(),'frozen':frozen,'source_time_shift_s':time_shift,'contact':contact,'status':status,'evidence':evidence})
   self.wait_until(self._joint_sim_stamp+.5)
   _,actual,status,evidence=self._snapshot()
  finally:
   with self._lock:hold=[self._joints[n] for n in JOINT_NAMES]
   self._publish_command(hold)
   if frozen:self._set_target_frozen('beverage_bottle',False)
   self._set_active(False)
  result={'label':label,'source_episode':index,'keep_collisions':keep,'hz':hz,'release':release,'onset_frame':onset,'threshold_frame':threshold,'initial_joints':initial,'source_initial_joints':states[0].tolist(),'reason':reason,'settle_before_lift':bool(getattr(self,'settle_before_lift',False)),'source_time_shift_s':time_shift,'sent':len(trace),'task_success':bool(status.get('task_success')),'physical_grasp_ok':bool(evidence.get('physical_grasp_ok')),'maximum_height':max((r['status'].get('object_position_m',[0,0,0])[2] for r in trace),default=0),'final_status':status,'final_evidence':evidence}
  (ROOT/(label+'_trace.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in trace));(ROOT/(label+'_result.json')).write_text(json.dumps(result,indent=2));print('TRIAL',json.dumps({k:v for k,v in result.items() if k not in ['final_status','final_evidence']}),flush=True);return result

def main():
 global ROOT
 parser=argparse.ArgumentParser();parser.add_argument('--suite',default='initial');parser.add_argument('--output',type=Path);opts=parser.parse_args();
 if opts.output is not None:
  ROOT=opts.output;ROOT.mkdir(parents=True,exist_ok=False)
 args=parse_args(['--checkpoint',str(DATA),'--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw']);args.training_manifest=None
 rclpy.init();node=DebugReplay(args);ex=MultiThreadedExecutor(num_threads=4);ex.add_node(node);thread=threading.Thread(target=ex.spin,daemon=True);thread.start();results=[]
 try:
  trials=[('approach_collision_on',True,10,'threshold',8.5),('approach_collision_off',False,10,'threshold',8.5),('legacy_release_threshold',False,10,'threshold',None),('legacy_release_onset',False,10,'onset',None),('interpolated_release_onset',False,60,'onset',None)]
  if opts.suite=='phase_checked':trials=[('checked_legacy_onset',False,10,'onset',None),('checked_interpolated_onset',False,60,'onset',None)]
  if opts.suite=='raw_teacher':
   node.data_root=BASE/'diagnostic_reconstructed_commands';trials=[('raw_teacher_60hz',True,60,'recorded',None),('raw_teacher_10hz',True,10,'recorded',None)]
  if opts.suite=='raw_ablation':
   node.data_root=BASE/'diagnostic_reconstructed_commands';trials=[('dense_repeat_1',True,60,'recorded',None),('dense_repeat_2',True,60,'recorded',None),('dense_repeat_3',True,60,'recorded',None),('dense_threshold_release',True,60,'threshold',None),('dense_onset_release',True,60,'onset',None),('sparse_interpolated',True,60,'recorded',None)]
  if opts.suite=='verified_replay':
   node.data_root=BASE/'diagnostic_sparse_commands';trials=[(f'verified_sparse_repeat_{i}',True,60,'recorded',None) for i in range(1,4)]
  if opts.suite=='settled_replay':
   node.data_root=BASE/'diagnostic_sparse_commands';node.settle_before_lift=True;trials=[(f'settled_sparse_repeat_{i}',True,60,'recorded',None) for i in range(1,4)]
  for label,keep,hz,release,duration in trials:
   if label=='sparse_interpolated':node.data_root=BASE/'diagnostic_sparse_commands'
   results.append(node.trial(0 if opts.suite in ['raw_teacher','raw_ablation','verified_replay','settled_replay'] else 89,label,keep,hz,release,duration));(ROOT/(opts.suite+'_results.json')).write_text(json.dumps(results,indent=2))
 finally:
  node._set_active(False);ex.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
