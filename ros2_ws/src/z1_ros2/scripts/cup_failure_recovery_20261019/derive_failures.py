import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from geometry import Geometry
R=Path('/home/lp531/smolvla_eval/act_cup_steps15_random30_20261019_202643/review')
ms=json.loads((R/'metrics.json').read_text());records=json.loads(Path('/home/lp531/pi05_lora_workspace/act_cup_cached_20261019/split_manifest.json').read_text())['source_records'];g=Geometry();refs=[]
for rec in records:
 if rec['source']!='original' or rec['split']!='train':continue
 p=Path(rec['source_path']);meta=json.loads((p/'episode_start.json').read_text());rows=[json.loads(x) for x in (p/'frames.jsonl').read_text().splitlines()];row=rows[rec['close_frame']];_,r=g.link_fk(row['observation.state']);refs.append((meta['scene_randomization']['object_positions_m']['cup'][:2],r,rec['episode_index']))
out=[]
for ep in [1,3,4,7,8,10,11,14,16,17,24,29]:
 m=next(x for x in ms if x['episode']==ep);rows=[json.loads(x) for x in (Path(m['path'])/'trace.jsonl').read_text().splitlines()]
 row=min(rows,key=lambda x:abs(x['simulation_elapsed_s']-m['closest_time']));c=row['contact_status'];rot=Rotation.from_quat(c['gripper_world_orientation_xyzw']);rel=np.array(c['relative_position_gripper_m']);world=rot.apply(np.array([.12083045,0,.01538985])-rel)
 _,rr,teacher=min(refs,key=lambda x:np.linalg.norm(np.array(x[0])-np.array(m['initial_position'][:2])))
 forward=rr[:,0][:2];forward=forward/np.linalg.norm(forward);left=np.array([-forward[1],forward[0]])
 raw=np.array([world[:2]@forward,world[:2]@left,world[2]])
 horizontal=raw[:2].copy();norm=np.linalg.norm(horizontal)
 if norm>.055:horizontal*=.055/norm
 bounded=[float(np.clip(horizontal[0],-.055,.004)),float(np.clip(horizontal[1],-.040,.040)),float(np.clip(raw[2],-.004,.025))]
 tilt=(Rotation.from_matrix(rr).inv()*rot).as_euler('xyz',degrees=True);tilt=np.clip(tilt,[-6,-6,-4],[6,6,4]).tolist()
 out.append(dict(eval_episode=ep,eval_seed=m['seed'],time_s=row['simulation_elapsed_s'],reference_training_episode=teacher,raw_world_offset_m=world.tolist(),raw_approach_offset_m=raw.tolist(),bounded_offset_m=bounded,relative_tool_rotation_deg=tilt,diagnostic_error_m=m['min_envelope_error_m']))
Path(__file__).with_name('failure_directions.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
