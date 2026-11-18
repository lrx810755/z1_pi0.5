import json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).parent;D=json.loads((R.parent/'comparison.json').read_text())
IDS=[2,3,4,5,7,8,9,12,13,14,15,18,20,22,23,24,27,29,30]
lo=np.array([.11283045,-.07,.008]);hi=np.array([.12883045,.07,.0227797]);rows=[]
for e in D['episodes']:
 if e['episode'] not in IDS:continue
 p=Path(e['path']);tr=[json.loads(l) for l in (p/'trace.jsonl').read_text().splitlines()];chunks=[json.loads(l) for l in (p/'prediction_chunks.jsonl').read_text().splitlines()]
 init=np.array(e['target_position']); candidates=[];firsttouch=None;firstshift=None;touches=[]
 for i,t in enumerate(tr):
  c=t['contact_status'];loc=c.get('relative_position_gripper_m');pos=c.get('object_position_m');tm=t['simulation_elapsed_s']
  touch=c.get('left_finger_contact_object')=='cup' or c.get('right_finger_contact_object')=='cup'
  if touch:
   touches.append(i)
   if firsttouch is None:firsttouch={'time':tm,'trace_index':i,'left':c.get('left_finger_contact_object'),'right':c.get('right_finger_contact_object'),'object_position':pos}
  shift=float(np.linalg.norm(np.array(pos[:2])-init[:2])) if pos else 0.
  if firstshift is None and shift>.005:firstshift={'time':tm,'trace_index':i,'shift_mm':shift*1000,'touch':touch,'tilt_deg':c.get('cylinder_tilt_deg')}
  if loc:
   loc=np.array(loc);clip=np.clip(loc,lo,hi);gap=loc-clip
   candidates.append((float(np.linalg.norm(gap)),i,gap,loc))
 gapnorm,i,gap,loc=min(candidates,key=lambda x:x[0]);t=tr[i];c=t['contact_status'];q=c['gripper_world_orientation_xyzw'];worldgap=Rotation.from_quat(q).apply(gap)
 tail=[t for t in tr if t['simulation_elapsed_s']>=50]
 worldposes=[np.array(t['contact_status']['gripper_world_position_m']) for t in tail if t['contact_status'].get('gripper_world_position_m')]
 pathlen=sum(float(np.linalg.norm(b-a)) for a,b in zip(worldposes,worldposes[1:]))
 prefixmax=max(max(x['gripper_prediction'][:15]) for x in chunks)
 discarded=[{'prediction':x.get('prediction'),'gripper_prediction':x['gripper_prediction'],'first_close_step_06':x.get('first_close_step_06')} for x in chunks if max(x['gripper_prediction'])>=.6 and max(x['gripper_prediction'][:15])<.6]
 end=tr[-1]['contact_status']
 row=dict(episode=e['episode'],path=e['path'],initial_position=e['target_position'],closest_capture=dict(time=t['simulation_elapsed_s'],trace_index=i,local_position_m=loc.tolist(),gap_to_box_mm=(gap*1000).tolist(),distance_mm=gapnorm*1000,world_move_to_box_mm=(worldgap*1000).tolist()),first_finger_contact=firsttouch,first_shift_over5mm=firstshift,max_cup_shift_mm=e['cup_shift_before_first_close_m']*1000,final_cup_position=end.get('object_position_m'),final_cup_tilt_deg=end.get('cylinder_tilt_deg'),final_relative_position=end.get('relative_position_gripper_m'),final_gripper_position=end.get('gripper_world_position_m'),finger_contact_samples=len(touches),last10s_gripper_path_length_m=pathlen,max_prediction_all16=e['max_predicted_close'],max_prediction_first15=prefixmax,discarded_close_predictions=discarded)
 rows.append(row)
 print(e['episode'],'gapmm',round(gapnorm*1000,2),'localgap',[round(x*1000,2) for x in gap],'worldgap',[round(x*1000,2) for x in worldgap],'t',round(t['simulation_elapsed_s'],2),'firsttouch',None if firsttouch is None else round(firsttouch['time'],2),'firstshift',None if firstshift is None else round(firstshift['time'],2),'shift',round(row['max_cup_shift_mm'],1),'endtilt',row['final_cup_tilt_deg'],'first15max',round(prefixmax,4))
(R/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
