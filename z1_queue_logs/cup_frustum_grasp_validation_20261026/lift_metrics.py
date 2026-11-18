"""Independent trace audit of grasp-only proof, separate from later box carry."""
import json,math
import numpy as np
from scipy.spatial.transform import Rotation

def proof_lift(path,baseline=0.):
 a=np.arange(64)*2*np.pi/64
 vertices=np.concatenate([np.column_stack([rad*np.cos(a),rad*np.sin(a),np.full(64,z)]) for rad,z in [(.018,0),(.0252,.061)]])
 window=[];longest=0.;first=None;max_clearance=-1.
 for line in path.open():
  row=json.loads(line);s=row['status'];t=row['sim_time'];p=s.get('object_position_m');oq=s.get('object_orientation_xyzw');gq=s.get('gripper_world_orientation_xyzw');local=s.get('relative_position_gripper_m')
  if not all([p,oq,gq,local]):window=[];continue
  obj=Rotation.from_quat(oq);rot=Rotation.from_quat(gq).inv()*obj
  clearance=float((obj.apply(vertices)+p)[:,2].min());max_clearance=max(max_clearance,clearance)
  forces=[s.get(k,0.) for k in ['left_finger_contact_impulse_ns','right_finger_contact_impulse_ns']];ratio=min(forces)/max(forces) if max(forces)>0 else 0
  valid=bool(p[2]-baseline>=.03 and clearance>=.02 and s.get('physical_grasp_ok') and s.get('bilateral_contact') and s.get('relative_pose_stable') and ratio>=.2 and not s.get('object_touching_table',True) and not s.get('frozen') and s.get('target_collisions_enabled'))
  if not valid:window=[];continue
  window.append((t,np.array(local),rot))
  while window and (np.linalg.norm(np.array(local)-window[0][1])>.006 or math.degrees((window[0][2].inv()*rot).magnitude())>5):window.pop(0)
  if window:
   duration=t-window[0][0];longest=max(longest,duration)
   if duration>=.4 and first is None:first=dict(start_sim_s=window[0][0],end_sim_s=t,minimum_clearance_m=clearance,root_z_m=p[2])
 return dict(proof_lift_pass=first is not None,first_stable_lift=first,longest_stable_lift_s=round(longest,4),maximum_geometric_clearance_m=max_clearance)
