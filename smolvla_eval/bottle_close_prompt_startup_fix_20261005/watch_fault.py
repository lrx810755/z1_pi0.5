import rclpy,cv2,time,json,urllib.request
from rclpy.node import Node
from sensor_msgs.msg import Image,JointState
from rclpy.qos import qos_profile_sensor_data
from z1_isaac_bridge.episode_recorder import image_to_bgr
rclpy.init();n=Node('collection_fault_image');latest={};cmd={};peak=[0]
def image(m):latest['image']=image_to_bgr(m)
def state(m):
 q=dict(zip(m.name,m.position));err=max([abs(cmd.get(k,v)-v) for k,v in q.items() if k!='jointGripper'] or [0])
 if err>peak[0] and 'image' in latest:
  peak[0]=err;cv2.imwrite('/home/lp531/smolvla_eval/bottle_close_prompt_startup_fix_20261005/fault.jpg',latest['image'])
  try:s=json.load(urllib.request.urlopen('http://127.0.0.1:8767/api/status',timeout=1))
  except Exception:s={}
  open('/home/lp531/smolvla_eval/bottle_close_prompt_startup_fix_20261005/fault.json','w').write(json.dumps({'error':err,'q':q,'cmd':cmd,'status':s},indent=2))
n.create_subscription(Image,'/global_camera/color/image_raw',image,qos_profile_sensor_data)
n.create_subscription(JointState,'/joint_states',state,qos_profile_sensor_data)
n.create_subscription(JointState,'/isaac_joint_commands',lambda m:cmd.update(zip(m.name,m.position)),10)
t=time.monotonic()
while time.monotonic()-t<30:rclpy.spin_once(n,timeout_sec=.02)
print('peak',peak[0]);n.destroy_node();rclpy.shutdown()
