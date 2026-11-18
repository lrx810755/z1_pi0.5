import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent;ms=json.loads((R/'metrics.json').read_text());out=[]
for m in ms:
 rows=[json.loads(x) for x in (Path(m['path'])/'trace.jsonl').read_text().splitlines()]
 times=np.array([q['simulation_elapsed_s'] for q in rows]);c=[q['contact_status'] for q in rows];rel=np.array([x['relative_position_gripper_m'] for x in c]);pos=np.array([x['object_position_m'] for x in c]);angle=np.array([x.get('contact_angle_valid',False) for x in c]);upright=np.array([x.get('cylinder_tilt_deg',180)<20 for x in c]);dt=np.diff(times,append=times[-1])
 mask=(times>=8)&upright&(pos[:,2]<.01)&(rel[:,0]>.10)&(rel[:,0]<.25)&(np.abs(rel[:,1])<.045)&(rel[:,2]>=.008)&(rel[:,2]<=.031)
 depthmask=mask&(rel[:,0]>.13883045)
 first=None
 if m['first_close_s'] is not None:
  idx=int(np.argmin(abs(times-m['first_close_s'])));first={'time':times[idx],'local':rel[idx].tolist(),'tilt':c[idx]['cylinder_tilt_deg'],'anglevalid':bool(angle[idx])}
  forward=Rotation.from_quat(c[idx]['gripper_world_orientation_xyzw']).apply([1,0,0]);later=int(np.argmin(abs(times-(times[idx]+1))))
  g0=np.array(c[idx]['gripper_world_position_m']);g1=np.array(c[later]['gripper_world_position_m']);first.update(depth_change_1s_m=float(rel[later,0]-rel[idx,0]),cup_motion_along_forward_1s_m=float((pos[later]-pos[idx])@forward),gripper_motion_along_forward_1s_m=float((g1-g0)@forward),local_after_1s=rel[later].tolist())
 last=mask&(times>50);mid=mask&(times>=10)&(times<=20)
 item={'episode':m['episode'],'success':m['final_box_success'],'depth_seconds_matching_yz_upright':float(dt[depthmask].sum()),'depth_seconds_with_approach_angle_valid':float(dt[depthmask&angle].sum()),'last10_local_median':np.median(rel[last],axis=0).tolist() if last.any() else None,'10to20_local_median':np.median(rel[mid],axis=0).tolist() if mid.any() else None,'first_close':first,'all_anglevalid_seconds':float(dt[angle].sum())};out.append(item)
 print(m['episode'],'depthsecs',round(item['depth_seconds_matching_yz_upright'],2),'angledepth',round(item['depth_seconds_with_approach_angle_valid'],2),'last',item['last10_local_median'],'early',item['10to20_local_median'],'first',first)
(R/'depth_probe.json').write_text(json.dumps(out,indent=2))
