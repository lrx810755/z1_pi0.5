"""Rigid-grasp targets and forward progress constraints, independent of ROS."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
class GuardViolation(RuntimeError):pass

def _pose(position,rotation):
 p=np.asarray(position,dtype=float);r=np.asarray(rotation,dtype=float)
 if p.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(r).all():raise GuardViolation('nonfinite or invalid pose')
 return p,r

def grasp_offset(ee_position,ee_rotation,object_position):
 p,r=_pose(ee_position,ee_rotation);obj=np.asarray(object_position,float)
 if obj.shape!=(3,) or not np.isfinite(obj).all():raise GuardViolation('invalid object position')
 return r.T@(obj-p)

def ee_target(object_target,desired_rotation,offset):
 p,r=_pose(object_target,desired_rotation);local=np.asarray(offset,float)
 if local.shape!=(3,) or not np.isfinite(local).all():raise GuardViolation('invalid grasp offset')
 return p-r@local

@dataclass
class ForwardGuard:
 start:object
 goal:object
 rotation:object
 phase:str
 reverse_tolerance:float=.003
 lateral_tolerance:float=.020
 orientation_tolerance:float=.035
 def __post_init__(self):
  self.start,self.rotation=_pose(self.start,self.rotation);self.goal=np.asarray(self.goal,float)
  if self.goal.shape!=(3,) or not np.isfinite(self.goal).all():raise GuardViolation('invalid goal')
  if self.phase not in ('lift','horizontal'):raise ValueError(self.phase)
  delta=self.goal[:2]-self.start[:2];self.distance=float(np.linalg.norm(delta));self.direction=delta/self.distance if self.distance>1e-9 else np.zeros(2);self.peak=0.
 def check(self,position,rotation):
  p,r=_pose(position,rotation);angle=float(Rotation.from_matrix(self.rotation.T@r).magnitude())
  if angle>self.orientation_tolerance:raise GuardViolation(f'attitude changed {angle:.5f}rad')
  if self.phase=='lift':
   progress=float(p[2]-self.start[2]);lateral=float(np.linalg.norm(p[:2]-self.start[:2]));limit=self.reverse_tolerance;length=float(self.goal[2]-self.start[2])
  else:
   displacement=p[:2]-self.start[:2];progress=float(displacement@self.direction);lateral=float(np.linalg.norm(displacement-progress*self.direction));limit=self.lateral_tolerance;length=self.distance
   if abs(p[2]-self.start[2])>self.reverse_tolerance:raise GuardViolation('horizontal carry left fixed-height corridor')
  if progress<self.peak-self.reverse_tolerance:raise GuardViolation(f'reverse progress {self.peak-progress:.5f}m')
  if progress>length+self.reverse_tolerance:raise GuardViolation('goal overshoot')
  if lateral>limit:raise GuardViolation(f'{self.phase} lateral displacement {lateral:.5f}m')
  self.peak=max(self.peak,progress)
  return dict(progress_m=progress,lateral_m=lateral,orientation_error_rad=angle)
