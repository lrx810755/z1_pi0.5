import rclpy,time,json,statistics
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,JointState
from pathlib import Path
rclpy.init();n=Node('cup20hz_stream_probe');samples={}
for name,topic,typ,qos in [('wrist','/left_wrist_camera/color/image_raw',Image,qos_profile_sensor_data),('global','/global_camera/color/image_raw',Image,qos_profile_sensor_data),('global_right','/global_right_camera/color/image_raw',Image,qos_profile_sensor_data),('state','/joint_states',JointState,20),('action','/isaac_joint_commands',JointState,20)]:
 samples[name]=[]
 def cb(msg,name=name):samples[name].append(msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)
 n.create_subscription(typ,topic,cb,qos)
end=time.monotonic()+20
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
result={}
for name,stamps in samples.items():
 gaps=[b-a for a,b in zip(stamps,stamps[1:])]
 result[name]=dict(messages=len(stamps),median_period_s=statistics.median(gaps) if gaps else None,max_gap_s=max(gaps) if gaps else None,unique=len(set(stamps)))
Path(__file__).with_name('stream_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));n.destroy_node();rclpy.shutdown()
