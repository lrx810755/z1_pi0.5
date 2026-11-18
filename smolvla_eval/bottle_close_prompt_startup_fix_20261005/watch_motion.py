import json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
rclpy.init();n=Node('collection_motion_diagnostic');rows=[]
def callback(kind,msg):
 rows.append({'wall':time.monotonic(),'kind':kind,'stamp':msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,'names':list(msg.name),'q':list(msg.position)})
n.create_subscription(JointState,'/joint_states',lambda m:callback('state',m),qos_profile_sensor_data)
n.create_subscription(JointState,'/isaac_joint_commands',lambda m:callback('command',m),10)
t=time.monotonic()
while time.monotonic()-t<30:rclpy.spin_once(n,timeout_sec=.1)
p=Path('/home/lp531/smolvla_eval/bottle_close_prompt_startup_fix_20261005/motion.jsonl');p.write_text(''.join(json.dumps(r)+'\n' for r in rows));print('rows',len(rows));n.destroy_node();rclpy.shutdown()
