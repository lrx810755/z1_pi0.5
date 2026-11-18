import time,json
from pathlib import Path
import numpy as np
import rclpy,tf2_ros
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
W=Path(__file__).resolve().parent
js={j.attrib['name']:j for j in ET.parse('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf').getroot().findall('joint')}
def fk(q):
 R=np.eye(3);p=np.zeros(3)
 for i,name in enumerate(['joint1','joint2','joint3','joint4','joint5','joint6','gripperStatorLink']):
  j=js[name];o=j.find('origin');p+=R@np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' ');R=R@Rotation.from_euler('xyz',np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
  if i<6:R=R@Rotation.from_rotvec(np.fromstring(j.find('axis').attrib['xyz'],sep=' ')*q[i]).as_matrix()
 return p,R
rclpy.init();n=Node('eval_link_motion_probe');b=tf2_ros.Buffer();l=tf2_ros.TransformListener(b,n);state={};out=(W/'after_live_probe.jsonl').open('a',buffering=1)
def joint(m):state['joint']=(m.header.stamp.sec+m.header.stamp.nanosec*1e-9,dict(zip(m.name,m.position)))
def objects(m):
 for z in m.transforms:
  if 'Bottle' in z.child_frame_id:state['object']=z
def status(m):
 if 'joint' not in state or 'object' not in state:return
 s=json.loads(m.data);stamp,q=state['joint'];q=[q['joint'+str(i+1)] for i in range(6)];p,R=fk(q);o=state['object'];ot=o.header.stamp.sec+o.header.stamp.nanosec*1e-9
 v=np.array([o.transform.translation.x,o.transform.translation.y,o.transform.translation.z]);oq=[o.transform.rotation.x,o.transform.rotation.y,o.transform.rotation.z,o.transform.rotation.w];center=v+Rotation.from_quat(oq).apply([0,0,.0785]);local=R.T@(center-p)
 r=dict(wall=time.time(),joint_stamp=stamp,object_stamp=ot,actual_q=q,fk_gripper_world=p.tolist(),fk_relative=local.tolist(),status_relative=s.get('relative_position_gripper_m'),status_object=s.get('object_position_m'),status_distance=s.get('gripper_object_distance_m'))
 for label,t in [('latest',Time()),('stamped',Time(nanoseconds=int(ot*1e9)))]:
  try:
   z=b.lookup_transform('world','gripperStator',t);r[label]=dict(stamp=z.header.stamp.sec+z.header.stamp.nanosec*1e-9,position=[z.transform.translation.x,z.transform.translation.y,z.transform.translation.z])
  except Exception as e:r[label]=str(e)
 out.write(json.dumps(r)+'\n')
n.create_subscription(JointState,'/joint_states',joint,20);n.create_subscription(TFMessage,'/isaac/object_transforms',objects,20);n.create_subscription(String,'/smolvla/status',status,20)
end=time.monotonic()+150
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.03)
out.close();n.destroy_node();rclpy.shutdown()
