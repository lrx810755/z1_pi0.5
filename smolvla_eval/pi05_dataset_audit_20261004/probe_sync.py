"""Read-only current telemetry probe; sends no robot commands."""
import json,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,JointState
rclpy.init();node=rclpy.create_node('pi05_dataset_audit_sync')
samples={'global':[],'wrist':[],'joint':[]};latest={};pairs=[]
def observe(name,msg):
    if name=='joint' and not set(['joint1','joint2','joint3','joint4','joint5','joint6','jointGripper']).issubset(msg.name):
        return
    stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    samples[name].append(stamp);latest[name]=stamp
for name,kind,topic in [('global',Image,'/global_camera/color/image_raw'),('wrist',Image,'/left_wrist_camera/color/image_raw'),('joint',JointState,'/joint_states')]:
    node.create_subscription(kind,topic,lambda msg,name=name:observe(name,msg),qos_profile_sensor_data)
end=time.monotonic()+10;next_sample=time.monotonic()
while time.monotonic()<end:
    rclpy.spin_once(node,timeout_sec=.01)
    if time.monotonic()>=next_sample and len(latest)==3:
        pairs.append(max(latest.values())-min(latest.values()));next_sample=time.monotonic()+.1
report={'scope':'10s passive current scene; samples latest messages at 10Hz. Not synchronized inference logs from the earlier moving trials.',
    'latest_triplet_skew_s_quantiles':np.quantile(pairs,[0,.5,.95,1]).tolist() if pairs else [],
    'cameras':{name:{'messages':len(vals),'gap_quantiles_s':np.quantile(np.diff(vals),[.5,.95,1]).tolist() if len(vals)>1 else []} for name,vals in samples.items()}}
Path(__file__).with_name('live_sync_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
node.destroy_node();rclpy.shutdown()
