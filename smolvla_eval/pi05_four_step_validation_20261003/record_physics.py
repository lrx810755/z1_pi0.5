import json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
ROOT=Path(__file__).resolve().parent
rclpy.init();node=Node('supplement_physics_provenance')
handle=(ROOT/'supplement_physics_history.jsonl').open('a')
def capture(topic,message):
 try: payload=json.loads(message.data)
 except ValueError: payload=message.data
 handle.write(json.dumps({'wall_time':time.time(),'topic':topic,'payload':payload},ensure_ascii=False)+'\n');handle.flush()
for topic in ['/smolvla/scene_grasp_command','/smolvla/scene_grasp_status','/smolvla/grasp_control_state','/smolvla/recording_status','/smolvla/status']:
 node.create_subscription(String,topic,lambda msg,topic=topic:capture(topic,msg),10)
def capture_joints(topic,message):
 handle.write(json.dumps({'wall_time':time.time(),'topic':topic,'payload':{'name':list(message.name),'position':list(message.position),'simulation_time':message.header.stamp.sec+message.header.stamp.nanosec*1e-9}})+'\n');handle.flush()
for topic in ['/joint_states','/isaac_joint_commands']:
 node.create_subscription(JointState,topic,lambda msg,topic=topic:capture_joints(topic,msg),qos_profile_sensor_data)
try:rclpy.spin(node)
finally:handle.close();node.destroy_node();rclpy.shutdown()
