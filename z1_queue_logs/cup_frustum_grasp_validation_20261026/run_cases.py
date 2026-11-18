"""Full-robot production grasp and proof-lift experiments; no ACT and no data collection."""
import os,json,time,threading,argparse,urllib.request,urllib.error,traceback,math
from pathlib import Path
import numpy as np,cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from sensor_msgs.msg import Image,JointState
from std_srvs.srv import Trigger
from moveit_msgs.msg import DisplayTrajectory
from rosidl_runtime_py.convert import message_to_ordereddict
R=Path(__file__).parent

def atomic(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)

def api(path,body=None,timeout=45):
 req=urllib.request.Request('http://127.0.0.1:8810'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=timeout) as f:return json.load(f)
 except urllib.error.HTTPError as exc:raise RuntimeError(exc.read().decode())

class Probe(Node):
 def __init__(self):
  super().__init__('cup_frustum_lift_probe',parameter_overrides=[Parameter('use_sim_time',value=True)])
  self.lock=threading.RLock();self.status={};self.scene={};self.assistant={};self.joints={};self.images={};self.active=None;self.rows=[];self.writers={}
  self.create_subscription(String,'/smolvla/status',lambda m:self.message('status',m),10)
  self.create_subscription(String,'/smolvla/scene_grasp_status',lambda m:self.message('scene',m),10)
  self.create_subscription(String,'/smolvla/pregrasp_status',lambda m:self.message('assistant',m),10)
  self.create_subscription(JointState,'/joint_states',self.joint,10)
  self.create_subscription(DisplayTrajectory,'/display_planned_path',self.display,10)
  for name,topic in [('global','/global_camera/color/image_raw'),('right','/global_right_camera/color/image_raw'),('wrist','/left_wrist_camera/color/image_raw')]:self.create_subscription(Image,topic,lambda m,n=name:self.camera(n,m),qos_profile_sensor_data)
  self._probe_clients={name:self.create_client(Trigger,'/smolvla/'+name) for name in ['auto_generate_pregrasp','auto_execute_pregrasp','auto_guided_grasp']}
  self.create_timer(.1,self.sample)
 def message(self,name,m):
  try:d=json.loads(m.data)
  except Exception:return
  with self.lock:
   setattr(self,name,d)
   if self.active:
    with (self.active/(name+'_events.jsonl')).open('a') as f:f.write(json.dumps(dict(sim_time=self.get_clock().now().nanoseconds/1e9,**d),ensure_ascii=False)+'\n')
 def display(self,m):
  with self.lock:
   if self.active:atomic(self.active/'planned_trajectory.json',message_to_ordereddict(m))
 def joint(self,m):
  with self.lock:self.joints=dict(zip(m.name,m.position))
 def camera(self,name,m):
  channels=4 if 'a' in m.encoding.lower() else 3
  if m.encoding.lower() not in ['rgb8','bgr8','rgba8','bgra8']:return
  im=np.frombuffer(m.data,dtype=np.uint8).reshape(m.height,m.step)[:,:m.width*channels].reshape(m.height,m.width,channels)
  if m.encoding.lower().startswith('rgb'):im=cv2.cvtColor(im,cv2.COLOR_RGBA2BGR if channels==4 else cv2.COLOR_RGB2BGR)
  elif channels==4:im=cv2.cvtColor(im,cv2.COLOR_BGRA2BGR)
  with self.lock:self.images[name]=im.copy()
 def sample(self):
  with self.lock:
   if not self.active:return
   row=dict(sim_time=self.get_clock().now().nanoseconds/1e9,status=self.status,scene=self.scene,assistant=self.assistant,joints=self.joints);self.rows.append(row)
   with (self.active/'trace.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
   for name,im in self.images.items():
    if name not in self.writers:
     writer=cv2.VideoWriter(str(self.active/(name+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),10,(im.shape[1],im.shape[0]));assert writer.isOpened();self.writers[name]=writer
    self.writers[name].write(im)
 def call(self,name,timeout=180):
  c=self._probe_clients[name]
  if not c.wait_for_service(timeout_sec=10):raise RuntimeError('Service missing: '+name)
  f=c.call_async(Trigger.Request());deadline=time.monotonic()+timeout
  while not f.done() and time.monotonic()<deadline:time.sleep(.05)
  if not f.done():raise RuntimeError('Service timeout: '+name)
  response=f.result();return dict(success=response.success,message=response.message)
 def start(self,d):
  with self.lock:self.active=d;self.rows=[];self.writers={}
 def finish(self):
  with self.lock:
   self.active=None
   for writer in self.writers.values():writer.release()
   self.writers={};return list(self.rows)

def run(node,case):
 d=R/'cases'/case['id'];d.mkdir(parents=True,exist_ok=True);assert not (d/'result.json').exists(),d
 atomic(R/'current_case.json',case);atomic(d/'case.json',case);steps=[];phase='reset';started=time.time();result={}
 try:
  response=api('/api/reset_episode',{},timeout=60);assert response.get('success'),response
  response=api('/api/target',{'target_object':'cup'});assert response.get('success',True),response
  response=api('/api/randomize_episode',{'curriculum_level':'full','target_grid':case['target_grid']},timeout=60);assert response.get('success'),response
  time.sleep(.5);node.start(d)
  for name,phase in [('auto_generate_pregrasp','planning'),('auto_execute_pregrasp','approach'),('auto_guided_grasp','grasp_and_lift')]:
   for attempt in range(int(case.get('planning_attempts',1)) if phase=='planning' else 1):
    response=node.call(name);steps.append(dict(stage=phase,attempt=attempt+1,**response));atomic(d/'operations.json',steps)
    if response['success']:break
    time.sleep(.2)
   if not response['success']:raise RuntimeError(response['message'])
   if phase=='planning':
    time.sleep(.2)
    if (d/'planned_trajectory.json').exists():atomic(d/'approach_reference.json',json.loads((d/'planned_trajectory.json').read_text()))
  phase='hold_verification';time.sleep(1.2)
  result['service_success']=True
 except Exception as exc:
  result.update(service_success=False,error=str(exc),failed_stage=phase)
  try:api('/api/force_stop',{},timeout=12)
  except Exception as cleanup:result['cleanup_error']=str(cleanup)
 finally:rows=node.finish()
 states=[r['status'] for r in rows];pos=[s['object_position_m'] for s in states if s.get('object_position_m')]
 last=states[-1] if states else {};initial=case['cup_position_m'][2];maximum=max((p[2] for p in pos),default=initial);ever=any(s.get('physical_grasp_ok') for s in states)
 result.update(case=case,steps=steps,elapsed_wall_s=time.time()-started,frames=len(rows),max_object_root_z_m=maximum,maximum_rise_m=maximum-initial,final_status=last,ever_physical_grasp=ever,table_contact=any(s.get('left_finger_touching_table') or s.get('right_finger_touching_table') for s in states))
 result['passed']=bool(result.get('service_success') and ever and last.get('physical_grasp_ok') and maximum-initial>=.03 and not last.get('object_touching_table',True))
 atomic(d/'result.json',result);print(json.dumps(dict(case=case['id'],passed=result['passed'],stage=result.get('failed_stage'),height=maximum,error=result.get('error')),ensure_ascii=False),flush=True);return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('cases',type=Path);args=parser.parse_args();cases=json.loads(args.cases.read_text());assert os.environ.get('ROS_DOMAIN_ID')=='94'
 deadline=time.monotonic()+240
 while time.monotonic()<deadline:
  try:
   s=api('/api/status',timeout=2)
   if s.get('connected') and all(s.get(k,{}).get('ready') for k in ['episode_reset','collision_scene','object_collisions']):break
  except Exception:pass
  time.sleep(2)
 else:raise RuntimeError('Experiment scene not ready')
 rclpy.init();node=Probe();ex=MultiThreadedExecutor(num_threads=6);ex.add_node(node);worker=threading.Thread(target=ex.spin,daemon=True);worker.start();results=[]
 try:
  time.sleep(1)
  for case in cases:results.append(run(node,case))
  atomic(R/(args.cases.stem+'_results.json'),results)
 finally:
  node.finish();ex.shutdown();worker.join(timeout=5);node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
