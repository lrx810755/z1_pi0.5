import json,time,threading,uuid,os,math,traceback
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy,qos_profile_sensor_data
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import String,Bool
from std_srvs.srv import Trigger,SetBool
from replay_core import command_prefix,validate_takeover
ROOT=Path(__file__).resolve().parent.parent
DATA=Path('/home/lp531/smolvla_data/z1_cup_policy_push_recovery20_20261028')
CURRENT=ROOT/'current.json'
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
class Replay(Node):
 def __init__(self):
  super().__init__('cup_push_replay',parameter_overrides=[Parameter('use_sim_time',value=True)])
  self.q=None;self.stamp=0;self.status={};self.random={};self.source={};self.scene={};self.history=[]
  transient=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
  self.create_subscription(JointState,'/joint_states',self.joint,qos_profile_sensor_data)
  self.create_subscription(String,'/smolvla/status',self.update_status,20)
  self.create_subscription(String,'/smolvla/scene_reset_status',lambda m:setattr(self,'random',json.loads(m.data)),transient)
  self.create_subscription(String,'/smolvla/scene_grasp_status',lambda m:setattr(self,'scene',json.loads(m.data)),20)
  self.pub=self.create_publisher(JointState,'/isaac_joint_commands',10);self.randpub=self.create_publisher(String,'/smolvla/scene_randomization_command',10);self.scenepub=self.create_publisher(String,'/smolvla/scene_grasp_command',10)
  self.active=self.create_publisher(Bool,'/smolvla/policy_control_active',transient);self.target=self.create_publisher(String,'/smolvla/target_object',10)
  self.service_clients={p:self.create_client(Trigger,p) for p in ['/smolvla/reset_episode','/smolvla/stop_motion','/smolvla/auto_generate_pregrasp','/smolvla/auto_execute_pregrasp','/smolvla/auto_guided_grasp','/smolvla/auto_guided_place','/smolvla/recording/start','/smolvla/recording/save','/smolvla/recording/discard']}
  self.servo=self.create_client(SetBool,'/smolvla/enable_servo_stream')
 def joint(self,m):
  q=dict(zip(m.name,m.position));names=[f'joint{i}' for i in range(1,7)]+['jointGripper']
  if all(x in q for x in names):self.q=[q[x] for x in names];self.stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 def update_status(self,m):
  self.status=json.loads(m.data);self.history.append((self.stamp,self.status.copy()));self.history=self.history[-1000:]
 def call(self,p,timeout=120):
  c=self.service_clients[p]
  if not c.wait_for_service(timeout_sec=10):raise RuntimeError('Missing service '+p)
  f=c.call_async(Trigger.Request());deadline=time.monotonic()+timeout
  while not f.done() and time.monotonic()<deadline:time.sleep(.02)
  if not f.done():raise RuntimeError('Service timeout '+p)
  x=f.result()
  if not x.success:raise RuntimeError(p+':'+x.message)
  return x.message
 def wait(self,test,timeout=30):
  end=time.monotonic()+timeout
  while time.monotonic()<end:
   if test():return
   time.sleep(.005)
  raise RuntimeError('Wait timeout')
 def simwait(self,duration):
  start=self.stamp;self.wait(lambda:self.stamp>=start+duration,60)
 def command(self,q):
  m=JointState();m.header.stamp=self.get_clock().now().to_msg();m.name=[f'joint{i}' for i in range(1,7)]+['jointGripper'];m.position=list(map(float,q));self.pub.publish(m)
 def reset(self,c):
  self.active.publish(Bool(data=False));self.target.publish(String(data='cup'));self.call('/smolvla/reset_episode')
  self.random={};self.randpub.publish(String(data=json.dumps(dict(seed=c['seed'],level='full',target_grid=None,settle_duration_s=2.,non_target_max_angular_speed_rad_s=.35))))
  self.wait(lambda:self.random.get('phase') in ['RANDOMIZED','ERROR'] and self.random.get('seed')==c['seed'],140)
  if self.random['phase']!='RANDOMIZED':raise RuntimeError('Randomization failed '+str(self.random))
  for name,expected in c['initial_five_object_layout'].items():
   actual=self.random['object_positions_m'][name]
   if np.linalg.norm(np.array(actual)-expected['position_m'])>.004:raise RuntimeError('Initial layout mismatch '+name)
  self.scene={};self.scenepub.publish(String(data=json.dumps(dict(command='unfreeze',target='cup'))));self.wait(lambda:self.scene.get('phase') in ['PHYSICS_READY','UNFROZEN'],10)
  if self.servo.wait_for_service(timeout_sec=1):
   f=self.servo.call_async(SetBool.Request(data=False));self.wait(f.done,5)
  self.call('/smolvla/stop_motion',10);self.active.publish(Bool(data=True));self.simwait(.2)
 def replay(self,c):
  rows=[json.loads(l) for l in Path(c['source_trace_path']).read_text().splitlines()];rows=command_prefix(rows,c['takeover_simulation_elapsed_s'])
  self.history=[];start=self.stamp;out=ROOT/'replay_attempts'/f"{c['candidate_id']}_{uuid.uuid4().hex[:8]}.jsonl";out.parent.mkdir(exist_ok=True)
  with out.open('w') as stream:
   for row in rows:
    goal=start+row['simulation_elapsed_s'];self.wait(lambda:self.stamp>=goal,10)
    s=self.status
    if s.get('cylinder_tilt_valid') and s.get('cylinder_tilt_deg',0)>10:raise RuntimeError('Replay tilted before takeover')
    if abs(s.get('object_position_m',[0,0,0])[2])>.015:raise RuntimeError('Replay cup left tabletop before takeover')
    self.command(row['command']);stream.write(json.dumps(dict(source_t=row['simulation_elapsed_s'],sim_stamp=self.stamp,joints=self.q,status=s))+'\n')
  # Immediately hold measured joints, then relinquish policy control to the expert controller.
  self.command(self.q);self.active.publish(Bool(data=False));self.call('/smolvla/stop_motion',10);self.simwait(.45)
  s=self.status.copy();validate_takeover(s,c['cup_initial_xyz_m'])
  recent=[v for t,v in self.history if t>=self.stamp-.4]
  for v in recent:validate_takeover(v,c['cup_initial_xyz_m'])
  expected=np.array(c['cup_xyz_m']);actual=np.array(s['object_position_m']);delta=float(np.linalg.norm(actual-expected))
  if delta>.025:raise RuntimeError(f'Replayed cup differs25mm from source ({delta:.4f}m)')
  if max(abs(np.array(self.q[:6])-np.array(c['joints_rad'][:6])))>.08:raise RuntimeError('Replayed joint pose differs from source')
  return dict(replay_trace=str(out),actual_takeover_status=s,actual_takeover_joints=self.q,source_cup_position_error_m=delta,actual_takeover_sim_stamp=self.stamp,source_initial_randomization=self.random)
 def collect(self,c,index,attempt):
  from z1_isaac_bridge.dataset_selection import grid_bin
  from z1_isaac_bridge.grasp_config import GRASP_CONFIG
  cfg=GRASP_CONFIG['scene_randomization'];grid=list(grid_bin(c['cup_initial_xyz_m'],(tuple(cfg['workspace_x_m']),tuple(cfg['workspace_y_m'])),tuple(cfg['levels']['full']['target_grid_shape'])))
  assert 0<=grid[0]<5 and 0<=grid[1]<4,grid
  self.reset(c);evidence=self.replay(c)
  split='validation' if tuple(grid) in [(1,1),(2,3)] else 'train';identity=c['candidate_id']+f"_takeover_{c['takeover_simulation_elapsed_s']:.6f}"
  context=dict(candidate=c,source_identity=identity,source_grid=grid,seed=c['seed'],task=dict(id=identity,kind='push_recovery',source_type='actual_policy_push_recovery',grid=grid,split=split),plan_hash='actual_policy_push_recovery19_v1',attempt_id=uuid.uuid4().hex)
  write(CURRENT,context);epath=ROOT/f"evidence_{context['attempt_id']}.json"
  meta=dict(task_id=identity,task=context['task'],source_type='actual_policy_push_recovery',source_identity=identity,source_grid=grid,seed=c['seed'],split=split,source_candidate=c,accepted=False,recovery_complete=False,recording_ready=True,expert_hold_verified=True,events=[dict(event='gripper_intent',time=self.stamp,state='open',target=-.9)],recovery_stages=[],recovery_start_sim_s=self.stamp,**evidence)
  write(epath,meta);recording=False
  try:
   self.call('/smolvla/stop_motion',10);self.call('/smolvla/recording/start',15);recording=True;self.simwait(.6)
   self.call('/smolvla/auto_generate_pregrasp',240)
   self.call('/smolvla/auto_execute_pregrasp',180)
   self.call('/smolvla/auto_guided_grasp',240)
   self.call('/smolvla/auto_guided_place',300)
   self.call('/smolvla/stop_motion',10);self.simwait(2.2)
   saved=self.call('/smolvla/recording/save',40);recording=False
   return dict(candidate_id=c['candidate_id'],message=saved,evidence=str(epath))
  except BaseException:
   if recording:
    try:self.call('/smolvla/recording/discard',10)
    except Exception:pass
   raise

def main():
 rclpy.init();node=Replay();executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);thread=threading.Thread(target=executor.spin,daemon=True);thread.start();node.wait(lambda:node.q is not None and bool(node.status),30)
 candidates=json.loads((ROOT/'candidates/candidates_manifest.json').read_text())['candidates'];saved=[];failures=[]
 for directory in DATA.glob('episode_[0-9]*'):
  m=json.loads((directory/'correction.json').read_text())
  if m.get('accepted'):saved.append(dict(candidate_id=m['source_candidate']['candidate_id'],episode=str(directory)))
 done={x['candidate_id'] for x in saved}
 try:
  for attempt in range(1,13):
   for index,c in enumerate(candidates):
    if c['candidate_id'] in done:continue
    write(ROOT/'status.json',dict(stage='replay_and_collect',candidate=c['candidate_id'],completed=len(saved),target=len(candidates),attempt=attempt))
    try:
     row=node.collect(c,index,attempt);saved.append(row);done.add(c['candidate_id']);write(ROOT/'saved_sources.json',saved)
     print('ACCEPTED',c['candidate_id'],len(saved),flush=True)
    except Exception as exc:
     failures.append(dict(candidate=c['candidate_id'],attempt=attempt,error=str(exc),traceback=traceback.format_exc()));write(ROOT/'rejected_attempts.json',failures)
     print('ATTEMPT_FAILED',c['candidate_id'],attempt,str(exc),flush=True)
   if len(done)==len(candidates):break
  if len(done)!=len(candidates):raise RuntimeError('Incomplete after12 passes: '+str([c['candidate_id'] for c in candidates if c['candidate_id'] not in done]))
  write(ROOT/'status.json',dict(stage='collection_finished_pending_independent_audit',completed=len(saved),target=len(candidates)))
 finally:
  node.active.publish(Bool(data=False));executor.shutdown();node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
