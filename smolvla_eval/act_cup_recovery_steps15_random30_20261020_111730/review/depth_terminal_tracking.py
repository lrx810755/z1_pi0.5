import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_failure_recovery_20261019')
from geometry import Geometry
g=Geometry();R=Path(__file__).resolve().parent;ms=json.loads((R/'metrics.json').read_text());out=[]
for ep in [7,9,11,12,13,14,19,22,23,25]:
 m=next(x for x in ms if x['episode']==ep);rows=[json.loads(x) for x in (Path(m['path'])/'trace.jsonl').read_text().splitlines()]
 for t in [20,40,55]:
  q=min(rows,key=lambda x:abs(x['simulation_elapsed_s']-t));c=q['contact_status'];actual,r=g.fk(q['joint_state']);cmd,cr=g.fk(q['command']);policy,pr=g.fk(q['policy_action']);forward=Rotation.from_quat(c['gripper_world_orientation_xyzw']).apply([1,0,0]);cup=np.array(c['object_position_m'])+np.array([0,0,.0305]);item=dict(episode=ep,time_s=float(q['simulation_elapsed_s']),local_m=c['relative_position_gripper_m'],tilt_deg=c['cylinder_tilt_deg'],immediate_command_axial_tracking_mm=float((cmd-actual)@forward*1000),model_action_axial_advance_mm=float((policy-actual)@forward*1000),policy_local_x_m=float(pr.T.dot(cup-policy)[0]))
  out.append(item)
  if t==55:print(item)
(R/'depth_terminal_tracking.json').write_text(json.dumps(out,indent=2))
