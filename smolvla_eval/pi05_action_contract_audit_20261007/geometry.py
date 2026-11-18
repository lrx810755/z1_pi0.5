from pathlib import Path
import json,importlib.util,struct,csv,sys
from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit','/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
mesh={}
for name in ('Stator','Mover'):
 d=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_description/meshes/collision/z1_Gripper'+name+'.STL').read_bytes();dt=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')]);v=np.unique(np.frombuffer(d,dtype=dt,count=struct.unpack_from('<I',d,80)[0],offset=84)['v'].reshape(-1,3),axis=0);mesh[name]=v[ConvexHull(v).vertices]
def height(q):
 p,r=audit.fk(q[:6]);fixed=mesh['Stator']@r.T+p;rot=Rotation.from_rotvec([0,q[-1],0]).as_matrix();moving=(mesh['Mover']@rot.T+[.049,0,0])@r.T+p
 return float(min(fixed[:,2].min(),moving[:,2].min()))