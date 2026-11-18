"""URDF FK and conservative finger mesh height; MoveIt checks full collisions."""
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
URDF=Path('/home/lp531/smolvla_eval/three_camera_dataset_strict_audit_20261006/robot.urdf')

class Geometry:
    def __init__(self):
        robot=ET.parse(URDF).getroot();self.chain=[];self.mesh={}
        for i in range(1,7):
            j=robot.find(f"joint[@name='joint{i}']");o=j.find('origin');t=np.eye(4)
            t[:3,3]=np.fromstring(o.get('xyz'),sep=' ')
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix()
            self.chain.append((t,np.fromstring(j.find('axis').get('xyz'),sep=' ')))
        for name in ['Stator','Mover']:
            data=(ROOT/'z1_description/meshes/collision'/f'z1_Gripper{name}.STL').read_bytes()
            dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
            vertices=np.unique(np.frombuffer(data,dtype=dtype,count=struct.unpack_from('<I',data,80)[0],offset=84)['v'].reshape(-1,3),axis=0)
            self.mesh[name]=vertices[ConvexHull(vertices).vertices]
    def link_fk(self,q):
        t=np.eye(4)
        for angle,(origin,axis) in zip(q,self.chain):
            r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(float(angle)*axis).as_matrix();t=t@origin@r
        return t[:3,3],t[:3,:3]
    def fk(self,q):
        p,r=self.link_fk(q)
        return p+r@np.array([.051,0,0]),r
    def height(self,q):
        p,r=self.fk(q[:6]);fixed=self.mesh['Stator']@r.T+p
        moving=(self.mesh['Mover']@Rotation.from_rotvec([0,q[6],0]).as_matrix().T+[.049,0,0])@r.T+p
        return float(min(fixed[:,2].min(),moving[:,2].min()))
    def check_path(self, points, floor=.006):
        previous=np.asarray(points[0],dtype=float)
        for point in points:
            point=np.asarray(point,dtype=float)
            if not np.isfinite(point).all():raise ValueError('nonfinite planned joint state')
            steps=max(1,int(np.ceil(np.max(np.abs(point-previous))/.01)))
            for alpha in np.linspace(0,1,steps+1):
                if self.height(previous+(point-previous)*alpha)<floor:
                    raise RuntimeError('纠正准备/恢复路径的夹爪桌面间隙不足 6 mm')
            previous=point
