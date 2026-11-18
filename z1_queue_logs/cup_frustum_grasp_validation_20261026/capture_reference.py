from pathlib import Path
import json,time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from moveit_msgs.msg import DisplayTrajectory
from rosidl_runtime_py.convert import message_to_ordereddict
rclpy.init();n=Node('capture_successful_grasp_approach');q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE);data=[]
s=n.create_subscription(DisplayTrajectory,'/display_planned_path',lambda m:data.append(message_to_ordereddict(m)),q);deadline=time.monotonic()+8
while not data and time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.2)
assert data,'No retained planned trajectory';r=Path(__file__).parent;v=dict(display=data[-1],candidate=json.loads((r/'cases/controlled_preload_0p5/candidate.json').read_text())['candidate'],cup_position_m=[.52,-.10,0.],source='controlled_preload_0p5');(r/'safe_reference.json').write_text(json.dumps(v,indent=2));print('Saved full approach trajectory:',len(data[-1]['trajectory'][0]['joint_trajectory']['points']),'points');n.destroy_node();rclpy.shutdown()
