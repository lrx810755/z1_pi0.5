from pathlib import Path
import json,numpy as np
out=Path(__file__).resolve().parent
rows=json.loads((out/'metrics.json').read_text())
profile=json.loads(Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/grasp_physics.json').read_text())['object_grasp_profiles']['beverage_bottle']
e=profile['preclose_envelope'];print('envelope',e)
results=[]
for r in rows:
 data=[json.loads(l) for l in (Path(r['path'])/'trace.jsonl').read_text().splitlines() if l.strip()]
 if not data:continue
 t=np.array([x['simulation_elapsed_s'] for x in data]);dt=np.r_[np.diff(t),0]
 rel=np.array([x['contact_status']['relative_position_gripper_m'] for x in data]);pos=np.array([x['contact_status']['object_position_m'] for x in data]);raw=np.array([x['policy_action'][-1] for x in data]);actual=np.array([x['joint_state'] for x in data]);env=(rel[:,0]>=e['x_min_m'])&(rel[:,0]<=e['x_max_m'])&(abs(rel[:,1])<=e['y_abs_max_m'])&(rel[:,2]>=e['z_min_m'])&(rel[:,2]<=e['z_max_m']);undisturbed=~np.maximum.accumulate(np.linalg.norm(pos-pos[0],axis=1)>.01);pre=~np.maximum.accumulate(raw>=.5)
 d={'model':r['model'],'run':r['run'],'envelope_seconds_before_close':float(dt[env&pre].sum()),'undisturbed_envelope_seconds':float(dt[env&pre&undisturbed].sum()),'raw_max_in_envelope':float(raw[env&pre].max()) if (env&pre).any() else None,'threshold_mismatches':sum((x['policy_action'][-1]>=.5)!=(x['execution_target'][-1]>=-.75) for x in data),'last10s_joint_range':np.ptp(actual[t>=t[-1]-10,:6],axis=0).tolist()}
 results.append(d)
 print(r['model'],r['run'][-6:],'env_s',round(d['envelope_seconds_before_close'],2),'undist_s',round(d['undisturbed_envelope_seconds'],2),'maxraw',d['raw_max_in_envelope'])
(out/'geometry_audit.json').write_text(json.dumps(results,indent=2))
