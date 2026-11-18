#!/usr/bin/env python3
"""Run a recording/transport probe on the isolated simulator domain only."""
import json, time, statistics
from pathlib import Path
from collections import deque
import cv2, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
ROOT=Path(__file__).parent
rclpy.init(); node=Node('three_camera_live_probe')
names=['joint1','joint2','joint3','joint4','joint5','joint6','jointGripper']
clock=[None]; joints=[None]; latest={}; stamps={n:[] for n in ['wrist','global','global_right']}; status={}; truth={}
node.create_subscription(Clock,'/clock',lambda m:clock.__setitem__(0,m.clock),qos_profile_sensor_data)
def state(m):
 d=dict(zip(m.name,m.position))
 if all(n in d for n in names):joints[0]=[d[n] for n in names]
node.create_subscription(JointState,'/joint_states',state,20)
def image(name,m):
 stamps[name].append(m.header.stamp.sec+m.header.stamp.nanosec*1e-9);latest[name]=m
for name,topic in [('wrist','/left_wrist_camera/color/image_raw'),('global','/global_camera/color/image_raw'),('global_right','/global_right_camera/color/image_raw')]:
 node.create_subscription(Image,topic,lambda m,n=name:image(n,m),qos_profile_sensor_data)
def rec(m):status.update(json.loads(m.data))
q=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
node.create_subscription(String,'/smolvla/recording_status',rec,q)
def transforms(m):
 for t in m.transforms:truth[t.child_frame_id]={'position':[t.transform.translation.x,t.transform.translation.y,t.transform.translation.z]}
node.create_subscription(TFMessage,'/isaac/object_transforms',transforms,qos_profile_sensor_data)
pub=node.create_publisher(JointState,'/isaac_joint_commands',20)
clients={n:node.create_client(Trigger,'/smolvla/recording/'+n) for n in ['start','discard']}
def spin(seconds,target=None):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  rclpy.spin_once(node,timeout_sec=.01)
  if clock[0] and joints[0]:
   m=JointState();m.header.stamp=clock[0];m.name=names;m.position=list(target if target is not None else joints[0]);pub.publish(m)
def call(name):
 assert clients[name].wait_for_service(timeout_sec=5),name
 f=clients[name].call_async(Trigger.Request()); end=time.monotonic()+10
 while not f.done() and time.monotonic()<end:spin(.02)
 assert f.done(),name
 response=f.result(); assert response.success,response.message
 return response.message
try:
 spin(5)
 assert all(latest.get(n) for n in stamps) and joints[0], 'missing streams'
 home=[0.,.002,-.002,0.,0.,0.,-.9]
 before=list(joints[0])
 for k in range(100):spin(.03,[a+(b-a)*(k+1)/100 for a,b in zip(before,home)])
 randomize=node.create_publisher(String,'/smolvla/scene_randomization_command',10)
 cmd=String();cmd.data=json.dumps({'seed':1046314014,'level':'full','target_grid':[1,2]})
 randomize.publish(cmd);spin(5,home)
 start=call('start');print('START',start,flush=True)
 poses=json.loads((ROOT/'probe_poses.json').read_text()); snapshots={}
 for name,target in [('home',joints[0]),*poses.items()]:
  before=list(joints[0])
  for k in range(80):spin(.04,[a+(b-a)*(k+1)/80 for a,b in zip(before,target)])
  spin(3,target)
  for camera,m in latest.items():
   a=np.asarray(m.data,dtype=np.uint8).reshape(m.height,m.step)[:,:m.width*3].reshape(m.height,m.width,3)
   if m.encoding=='rgb8':a=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
   cv2.imwrite(str(ROOT/f'{name}_{camera}.jpg'),a)
  snapshots[name]={'measured_joints':joints[0], 'requested_joints':target,'truth':dict(truth),'recorder':dict(status)}
 spin(30,poses['right90'])
 paths=list((ROOT/'raw_probe_full_stack').glob('.recording_*/frames.jsonl'));assert len(paths)==1,paths
 frames=[json.loads(line) for line in paths[0].read_text().splitlines()]
 report={'frames':len(frames),'recording_status':dict(status),'snapshots':snapshots,'transport':{}}
 for n,ts in stamps.items():
  gaps=[b-a for a,b in zip(ts,ts[1:])]
  report['transport'][n]={'received':len(ts),'max_stamp_gap_s':max(gaps),'gaps_over_90ms':sum(g>.09 for g in gaps),'nonmonotonic':sum(g<=0 for g in gaps)}
 spreads=[max([r['joint_state_stamp'],r['wrist_image_stamp'],r['global_image_stamp'],r['global_right_image_stamp']])-min([r['joint_state_stamp'],r['wrist_image_stamp'],r['global_image_stamp'],r['global_right_image_stamp']]) for r in frames]
 report['max_recorded_spread_s']=max(spreads,default=None)
 report['p95_recorded_spread_s']=float(np.quantile(spreads,.95)) if spreads else None
 report['max_action_age_s']=max((r['action_age_s'] for r in frames),default=None)
 report['pass']=len(frames)>=100 and max(spreads)<=.045000001 and not status.get('sync_failure') and all(v['gaps_over_90ms']==0 for v in report['transport'].values())
 (ROOT/'live_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='snapshots'},indent=2),flush=True)
 # Probe is deliberately not a successful grasp demonstration.
 call('discard')
 assert report['pass'], 'live recording failed validation; see live_report.json'
finally:node.destroy_node();rclpy.shutdown()
