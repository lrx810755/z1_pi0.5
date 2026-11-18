"""Read-only numerical IK diagnostic; no ROS commands or collision exemptions."""
import sys,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_correction_fragments_20261019')
from geometry import Geometry,URDF
import xacro,xml.etree.ElementTree as ET
import z1_isaac_bridge.grasp_assistant as m
root=Path(__file__).parent
plan=json.loads((root/'collection_plan.json').read_text())
g=Geometry();robot=ET.fromstring(xacro.process_file(str(URDF)).toxml())
bounds=np.array([[float(robot.find(f"joint[@name='joint{i}']/limit").get(k)) for i in range(1,7)] for k in ['lower','upper']])
rng=np.random.default_rng(42)
output=[]
for task in plan['tasks']:
 if task['id'] not in ['new_correction_episode_000014','new_correction_episode_000023','new_correction_episode_000026','new_correction_episode_000027','new_correction_episode_000032','new_correction_episode_000033']:continue
 cfg=m.GRASP_PROFILES['cup'];ref=cfg['calibrated_grasp_reference']
 obj=np.array(task['scene_randomization']['object_positions_m']['cup']);obj[2]+=m.GRASP_CONFIG['object_center_z_offsets_m']['cup']
 seed=np.array(task['original_correction']['start_joint_state'][:6]);seed[5]*=-1
 for az in range(-90,91,5):
  pitch=math.radians(-6);a=math.radians(az)
  q=m.orientation_for_approach((math.cos(pitch)*math.cos(a),math.cos(pitch)*math.sin(a),math.sin(pitch)),-90)
  local=m._bottle_reference_for_orientation(ref['object_in_gripper_local_m'],q,ref['captured_object_center_minus_stator_world_z_m'])
  p=np.array(m._ee_position_for_object_in_stator(obj,q,(.051,0,0),(0,0,0,1),local))
  forward=np.array([math.cos(a),math.sin(a),0]);left=np.array([-math.sin(a),math.cos(a),0])
  dx,dy,dz=task['task']['offset_approach_m'];bad=p+(dx-task['task'].get('preparation_retreat_m',.004))*forward+dy*left+np.array([0,0,dz])
  badq=(R.from_quat(q)*R.from_euler('xyz',task['task']['rotation_tool_deg'],degrees=True)).as_quat()
  results=[]
  for label,goal,goalq in [('grasp',p,q),('deviation',bad,badq)]:
   def residual(j):
    pos,rot=g.link_fk(j)
    return np.r_[(pos-goal)/.002,(R.from_quat(goalq).inv()*R.from_matrix(rot)).as_rotvec()/.025]
   best=None
   for initial in [seed,(bounds[0]+bounds[1])/2,*rng.uniform(bounds[0],bounds[1],size=(2,6))]:
    fit=least_squares(residual,np.clip(initial,bounds[0]+1e-6,bounds[1]-1e-6),bounds=bounds,max_nfev=100)
    score=float(np.linalg.norm(fit.fun))
    if best is None or score<best['score']:best=dict(score=score,joints=fit.x.tolist())
    if score<.05:break
   results.append(best)
  if max(x['score'] for x in results)<1:
   output.append(dict(task_id=task['id'],azimuth=az,results=results));print(task['id'],az,[round(x['score'],3) for x in results],flush=True)
(root/'ik_diagnostic_large_offsets.json').write_text(json.dumps(output,indent=2))
