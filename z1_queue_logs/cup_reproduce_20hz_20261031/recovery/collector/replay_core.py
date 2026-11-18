import math
def command_prefix(rows,end):
 out=[];last=-math.inf
 for r in rows:
  t=r['simulation_elapsed_s']
  if t<last:raise ValueError('Nonmonotonic source trace')
  last=t
  if t<=end+1e-8:out.append(r)
 return out
def validate_takeover(s,initial):
 if s.get('cylinder_tilt_valid') is not True or s.get('cylinder_tilt_deg',180)>10:raise ValueError('Cup tilted')
 p=s.get('object_position_m',[])
 if len(p)!=3 or abs(p[2])>.008 or s.get('object_touching_table') is not True:raise ValueError('Cup not supported on table')
 if s.get('object_speed_m_s',1)>.02:raise ValueError('Cup not settled')
 if s.get('frozen') is not False or s.get('target_collisions_enabled') is not True:raise ValueError('Cup physics altered')
 if s.get('gripper_position_rad',0)>-.75 or s.get('physical_grasp_ok'):raise ValueError('Not open ungrasped takeover')
 if math.hypot(p[0]-initial[0],p[1]-initial[1])<.009:raise ValueError('No measurable push')
 return True
