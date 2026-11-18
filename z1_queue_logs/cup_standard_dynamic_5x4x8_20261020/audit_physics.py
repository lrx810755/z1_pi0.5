import json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from std_msgs.msg import String
r=Path(__file__).resolve().parent
rclpy.init()
n=Node('cup_dynamic_collection_audit')
phase={}
def stage(m):
 global phase
 phase=json.loads(m.data)
def status(m):
 d=json.loads(m.data)
 fields=['target_object','frozen','target_collisions_enabled','object_position_m','object_pose_stamp_s','phase','gripper_commanded_closed','holding_observed','task_success','released']
 out={k:d.get(k) for k in fields};out['wall_time']=time.time();out['pregrasp']=phase
 with (r/'physics_audit.jsonl').open('a') as f:f.write(json.dumps(out,ensure_ascii=False)+'\n')
q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
n.create_subscription(String,'/smolvla/pregrasp_status',stage,q)
n.create_subscription(String,'/smolvla/status',status,10)
rclpy.spin(n)
