#!/usr/bin/env python3
"""Execute production gripper methods against the actual isolated simulator."""
import json,time,threading
from pathlib import Path
from types import SimpleNamespace
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
from z1_isaac_bridge.grasp_assistant import GraspAssistant
ROOT=Path(__file__).parent
rclpy.init();node=Node('actual_gripper_open_probe');node.set_parameters([Parameter('use_sim_time',value=True)])
group=ReentrantCallbackGroup();status={};commands=[];events=[]
ctx=SimpleNamespace(_lock=threading.RLock(),_joint_positions={},_gripper_feedback=None,
    gripper_open=-.9,gripper_closed=0.,gripper_tolerance=.03,gripper_open_duration=2.,
    gripper_close_duration=1.5,gripper_release_threshold=-.75,gripper_grasp_threshold=-.75,
    _active_goal_handle=None,_persistent_gripper_goal_handle=None,_stop_requested=False,
    get_clock=node.get_clock,get_logger=node.get_logger)
for name in ['_joint_callback','_execute_gripper','_execute_gripper_target','_cancel_persistent_gripper_hold',
             '_raise_if_stop_requested','_send_action_goal','_wait_for_joint_value']:
 setattr(ctx,name,getattr(GraspAssistant,name).__get__(ctx))
ctx._wait_future=GraspAssistant._wait_future
ctx._set_status=lambda *args:events.append({'sim_time':node.get_clock().now().nanoseconds*1e-9,'status':args})
ctx._gripper_client=ActionClient(node,FollowJointTrajectory,'/gripper_trajectory_controller/follow_joint_trajectory',callback_group=group)
node.create_subscription(JointState,'/joint_states',ctx._joint_callback,20,callback_group=group)
def command(m):commands.append({'stamp':m.header.stamp.sec+m.header.stamp.nanosec*1e-9,'names':list(m.name),'positions':list(m.position)})
node.create_subscription(JointState,'/isaac_joint_commands',command,20,callback_group=group)
q=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
node.create_subscription(String,'/smolvla/recording_status',lambda m:status.update(json.loads(m.data)),q,callback_group=group)
clients={n:node.create_client(Trigger,'/smolvla/recording/'+n,callback_group=group) for n in ['start','discard']}
executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
def call(n):
 assert clients[n].wait_for_service(timeout_sec=5),n
 response=ctx._wait_future(clients[n].call_async(Trigger.Request()),10)
 assert response and response.success,response.message if response else n
 return response.message
def run(label,operation):
 before=node.get_clock().now().nanoseconds*1e-9
 with ctx._lock:
  actual_before=ctx._joint_positions.get('jointGripper');feedback=ctx._gripper_feedback
 eligible=(actual_before is not None and abs(actual_before+.9)<=.01 and feedback is not None
           and 0<=time.monotonic()-feedback[0]<=.2 and -.025<=before-feedback[1]<=.1)
 start=time.monotonic();actual=operation()
 row={'test':label,'duration_wall_s':time.monotonic()-start,'duration_sim_s':node.get_clock().now().nanoseconds*1e-9-before,'actual_gripper_rad':actual,'actual_before_rad':actual_before,'eligible_for_short_hold':eligible}
 print(json.dumps(row),flush=True);return row
try:
 until=time.monotonic()+10
 while (not ctx._gripper_feedback or not status.get('ready')) and time.monotonic()<until:time.sleep(.02)
 assert ctx._gripper_feedback and status.get('ready'),status
 ctx._execute_gripper('open')
 print(call('start'),flush=True)
 results=[run('already_open',lambda:ctx._execute_gripper('open')) for _ in range(6)]
 run('set_partial_open_for_control',lambda:ctx._execute_gripper_target('close',-.6))
 results.append(run('partially_open_normal_open',lambda:ctx._execute_gripper('open')))
 results.append(run('already_open_after_normal_open',lambda:ctx._execute_gripper('open')))
 path=next((ROOT/'raw_probe').glob('.recording_*/frames.jsonl'))
 frames=[json.loads(line) for line in path.read_text().splitlines()]
 (ROOT/'recorded_frames_audit.jsonl').write_text('\n'.join(json.dumps(r) for r in frames)+'\n')
 spreads=[max(r[k] for k in ['joint_state_stamp','wrist_image_stamp','global_image_stamp','global_right_image_stamp'])-min(r[k] for k in ['joint_state_stamp','wrist_image_stamp','global_image_stamp','global_right_image_stamp']) for r in frames]
 report={'results':results,'recorded_frames':len(frames),'max_recorded_spread_s':max(spreads),'recorder_status':dict(status),'events':events}
 fast=[r['duration_sim_s'] for r in results if r['test'].startswith('already_open') and r['eligible_for_short_hold']]
 fallback=[r['duration_sim_s'] for r in results if r['test'].startswith('already_open') and not r['eligible_for_short_hold']]
 normal=results[-2]['duration_sim_s']
 report['pass']=len(fast)>=5 and max(fast)<.3 and all(t>=1.9 for t in fallback) and normal>=1.9 and not status.get('sync_failure') and max(spreads)<=.045000001
 (ROOT/'live_report.json').write_text(json.dumps(report,indent=2))
 (ROOT/'commands.jsonl').write_text('\n'.join(json.dumps(r) for r in commands)+'\n')
 print(call('discard'),flush=True)
 assert report['pass'],report
finally:
 executor.shutdown(timeout_sec=3);thread.join(timeout=3);node.destroy_node();rclpy.shutdown()
