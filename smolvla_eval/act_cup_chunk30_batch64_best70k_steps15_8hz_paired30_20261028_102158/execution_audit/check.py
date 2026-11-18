import json, pathlib, xml.etree.ElementTree as ET
import numpy as np
R=pathlib.Path(__file__).resolve().parent.parent
URDF=pathlib.Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf')
J={j.attrib['name']:j for j in ET.parse(URDF).getroot().findall('joint')}
def fk(q,point=(0,0,0)):
 q=np.asarray(q);n=len(q);rot=np.tile(np.eye(3),(n,1,1));pos=np.zeros((n,3))
 for k,name in enumerate(['joint1','joint2','joint3','joint4','joint5','joint6','gripperStatorLink']):
  j=J[name];o=j.find('origin');assert o.attrib['rpy']=='0 0 0'
  pos+=np.einsum('nij,j->ni',rot,np.fromstring(o.attrib['xyz'],sep=' '))
  if k<6:
   axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ');x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
   a=q[:,k];rr=np.eye(3)[None]+np.sin(a)[:,None,None]*K+(1-np.cos(a))[:,None,None]*(K@K)
   rot=rot@rr
 return pos+np.einsum('nij,j->ni',rot,np.asarray(point)),rot
if __name__=='__main__':
 e=json.loads((R/'结果与失败分类.json').read_text())['episodes'][0]
 t=[json.loads(x) for x in (pathlib.Path(e['source_result']).parent/'trace.jsonl').read_text().splitlines()]
 q=np.array([x['joint_state'][:6] for x in t]);p,rot=fk(q);tf=np.array([x['contact_status']['gripper_world_position_m'] for x in t]);err=np.linalg.norm(p-tf,axis=1)
 print('FK TF err m',np.percentile(err,[0,50,95,100]),'initial',p[0],tf[0])
