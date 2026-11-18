import json,time,threading
from pathlib import Path
import rclpy
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String,Bool
from rosgraph_msgs.msg import Clock
from z1_isaac_bridge.trajectory_bridge import TrajectoryBridge,ALL_JOINTS
from z1_isaac_bridge.recording_sync import RecordingSynchronizer
rclpy.init()
bridge=TrajectoryBridge();bridge.set_parameters([Parameter('use_sim_time',value=True)]);observer=rclpy.create_node('isolated_hold_observer',parameter_overrides=[Parameter('use_sim_time',value=True)])
q=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
cp=observer.create_publisher(Clock,'/clock',10);rp=observer.create_publisher(String,'/smolvla/recording_status',q)
sp=observer.create_publisher(JointState,'/joint_states',10);contact=observer.create_publisher(String,'/smolvla/status',10)
pp=observer.create_publisher(Bool,'/smolvla/policy_control_active',10)
messages=[]
sub=observer.create_subscription(JointState,'/isaac_joint_commands',messages.append,10)
executor=SingleThreadedExecutor();executor.add_node(bridge);executor.add_node(observer)
index=6000

def run(recording,seconds):
 global index
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  index+=1;t=index/60
  clock=Clock();clock.clock.sec=int(t);clock.clock.nanosec=round((t-int(t))*1e9);cp.publish(clock)
  state=JointState();state.header.stamp=clock.clock;state.name=list(ALL_JOINTS);state.position=[0.0]*len(ALL_JOINTS);sp.publish(state)
  contact.publish(String(data=json.dumps(dict(contact_report_available=True,left_finger_touching_table=False,right_finger_touching_table=False))))
  rp.publish(String(data=json.dumps(dict(recording=recording))))
  tick_end=time.monotonic()+.016
  while time.monotonic()<tick_end:executor.spin_once(timeout_sec=.001)
run(False,.4)
with bridge._state_lock:bridge._command=dict.fromkeys(ALL_JOINTS,.01)
start=len(messages);run(True,1.2);held=messages[start:]
assert len(held)>20,len(held)
assert all(list(m.position)==[.01]*len(ALL_JOINTS) for m in held)
stamps=[m.header.stamp.sec+m.header.stamp.nanosec*1e-9 for m in held]
assert all(100 < t < 110 for t in stamps)
gaps=[b-a for a,b in zip(stamps,stamps[1:])];assert max(gaps)<.1,gaps
sync=RecordingSynchronizer((),action_age=.4)
for m,t in zip(held,stamps):
 sync.push('action',t,m.position);sync.push('state',t,[0.0]*7);sync.push('status',t,{})
 assert sync.select(t)['action'][0]==t
run(False,.15);stopped=len(messages);run(False,.3);assert len(messages)==stopped
# Ownership suppression is checked after queued messages have drained.
checks={}
for owner in ['servo','policy','reset','arm','gripper']:
 if owner=='servo':bridge._servo_stream_enabled=True
 elif owner=='policy':bridge._policy_control_active=True
 elif owner=='reset':bridge._scene_reset_active=True
 else:bridge._controller_locks[owner].acquire()
 run(True,.12);before=len(messages);run(True,.15);checks[owner]=len(messages)-before
 assert checks[owner]==0,(owner,checks)
 if owner=='servo':bridge._servo_stream_enabled=False
 elif owner=='policy':bridge._policy_control_active=False
 elif owner=='reset':bridge._scene_reset_active=False
 else:bridge._controller_locks[owner].release()
result=dict(domain=89,held_messages=len(held),max_gap_s=max(gaps),held_target=.01,stop_recording_suppresses=True,ownership_extra_messages=checks)
Path(__file__).with_name('ros_smoke_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
executor.remove_node(bridge);executor.remove_node(observer);bridge.destroy_node();observer.destroy_node();rclpy.shutdown()
