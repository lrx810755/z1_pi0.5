"""Preplan all recovery motion; prefer a direct continuous return to the grasp pose."""
import numpy as np
from scipy.spatial.transform import Rotation

def routes(position,orientation,object_position,task):
 p=np.array(position,dtype=float);q=list(orientation)
 forward=Rotation.from_quat(q).apply([1.,0.,0.])[:2];forward/=np.linalg.norm(forward);left=np.array([-forward[1],forward[0]])
 dx,dy,dz=task['offset_approach_m'];retreat=float(task.get('preparation_retreat_m',0))
 bad=p.copy();bad[:2]+=(dx-retreat)*forward+dy*left;bad[2]+=dz
 if task['kind']=='post_shift_realign':
  bad=p-np.array(task['actual_shift_world_m']);bad[:2]-=.012*forward;bad[2]+=.008
 bad_q=(Rotation.from_quat(q)*Rotation.from_euler('xyz',task['rotation_tool_deg'],degrees=True)).as_quat().tolist()
 prep=[dict(stage='prepare_deviation',position=bad.tolist(),orientation=bad_q,linear=False)]
 for strategy in ['direct','raised']:
  if strategy=='direct':
   recovery=[dict(stage='recover_align_approach',position=p.tolist(),orientation=q,linear=True)]
  else:
   z=max(p[2],bad[2])+.018
   recovery=[dict(stage='recover_clearance',position=[bad[0],bad[1],z],orientation=q,linear=False),
             dict(stage='recover_approach',position=p.tolist(),orientation=q,linear=True)]
  yield dict(anchor_position=p.tolist(),bad_position=bad.tolist(),bad_orientation=bad_q,retreat_m=retreat,
             strategy=strategy,preparation=prep,recovery=recovery)

def choose_route(candidates,plan_route):
 failures=[]
 for candidate in candidates:
  try:return candidate,plan_route(candidate),failures
  except RuntimeError as exc:failures.append(str(exc))
 raise RuntimeError('纠正全路径预检未通过: '+' | '.join(failures[-3:]))

def trajectory_points(trajectory,initial):
 points=[list(initial)]
 for point in trajectory.points:
  q=list(initial)
  for name,value in zip(trajectory.joint_names,point.positions):
   if name in ['joint1','joint2','joint3','joint4','joint5','joint6']:q[int(name[-1])-1]=float(value)
   elif name=='jointGripper':q[6]=float(value)
  points.append(q)
 return points
