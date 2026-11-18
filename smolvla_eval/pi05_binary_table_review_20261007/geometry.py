"""URDF mesh height comparison in robot base coordinates, not a physics replay."""
from pathlib import Path
import json,importlib.util,struct
import numpy as np
from scipy.spatial.transform import Rotation
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit','/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006/audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
meshes={}
for name in ['Stator','Mover']:
 data=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_description/meshes/collision/z1_Gripper'+name+'.STL').read_bytes()
 dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')]);n=struct.unpack_from('<I',data,80)[0]
 meshes[name]=np.unique(np.frombuffer(data,dtype=dtype,count=n,offset=84)['vertices'].reshape(-1,3),axis=0)
def heights(q):
 pos,r=a.fk(q[:6]);v=meshes['Stator']@r.T+pos
 rm=Rotation.from_rotvec(np.array([0,q[-1],0])).as_matrix();vm=(meshes['Mover']@rm.T+np.array([.049,0,0]))@r.T+pos
 return [float(v[:,2].min()),float(vm[:,2].min())]
d=json.loads((p/'analysis.json').read_text())
for e in d['episodes']:
 last=e['last'];raw=last['pred'][:6]+[last['cmd'][-1]];clipped=np.clip(raw,a.limits[:,0],a.limits[:,1]);e['last_gripper_mesh_min_height_base_m']={'actual':heights(last['q']),'command':heights(last['cmd']),'raw_model_with_mapped_gripper':heights(raw),'clipped_model':heights(clipped)}
 print(e['run'][-6:],e['last_gripper_mesh_min_height_base_m'])
(p/'analysis.json').write_text(json.dumps(d,indent=2)+'\n')
