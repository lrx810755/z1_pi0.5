import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_failure_recovery_20261019')
from geometry import Geometry
g=Geometry();R=Path(__file__).resolve().parent;ms=json.loads((R/'metrics.json').read_text());out=[]
for ep in [2,5,9,11,13,19,22,23,25]:
 m=next(x for x in ms if x['episode']==ep);rows=[json.loads(x) for x in (Path(m['path'])/'trace.jsonl').read_text().splitlines()]
 t=m['first_close_s'] if m['first_close_s'] is not None else m['closest_time']
 for dt in [-.5,0,1]:
  q=min(rows,key=lambda x:abs(x['simulation_elapsed_s']-(t+dt)));c=q['contact_status'];p=np.array(c['gripper_world_position_m']);rot=Rotation.from_quat(c['gripper_world_orientation_xyzw']);forward=rot.apply([1,0,0]);actual,rr=g.fk(q['joint_state']);cmd,cr=g.fk(q['command']);policy,pr=g.fk(q['policy_action']);target,tr=g.fk(q['execution_target']);cup=np.array(c['object_position_m'])+np.array([0,0,.0305]);rel=rot.inv().apply(cup-p)
  item=dict(episode=ep,time_s=float(q['simulation_elapsed_s']),local=c['relative_position_gripper_m'],cup_tilt=c['cylinder_tilt_deg'],actual_fk_vs_logged_mm=float(np.linalg.norm(actual-p)*1000),local_formula_error_mm=float(np.linalg.norm(rel-c['relative_position_gripper_m'])*1000),immediate_command_minus_actual_axial_mm=float((cmd-actual)@forward*1000),policy_target_minus_actual_axial_mm=float((policy-actual)@forward*1000),command_local_x_m=float(cr.T.dot(cup-cmd)[0]),policy_local_x_m=float(pr.T.dot(cup-policy)[0]),arm_joint_error_max=float(np.max(np.abs(np.array(q['command'][:6])-q['joint_state'][:6]))))
  out.append(item);print(json.dumps(item))
(R/'depth_tracking.json').write_text(json.dumps(out,indent=2))
