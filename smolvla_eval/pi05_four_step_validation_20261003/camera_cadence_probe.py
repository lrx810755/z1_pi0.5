import json,time,statistics
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
rclpy.init();n=Node('camera_cadence_probe');samples={'global':[],'wrist':[]}
for name,topic in [('global','/global_camera/color/image_raw'),('wrist','/left_wrist_camera/color/image_raw')]:
 n.create_subscription(Image,topic,lambda msg,name=name:samples[name].append(msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9),qos_profile_sensor_data)
end=time.monotonic()+15
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
result={}
for name,rows in samples.items():
 gaps=[b-a for a,b in zip(rows,rows[1:])];result[name]={'frames':len(rows),'median_gap_s':statistics.median(gaps) if gaps else None,'max_gap_s':max(gaps) if gaps else None,'gaps_over_90ms':sum(x>.09 for x in gaps)}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2));print(result);n.destroy_node();rclpy.shutdown()
