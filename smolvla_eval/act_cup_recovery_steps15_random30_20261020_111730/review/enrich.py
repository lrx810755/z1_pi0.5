import json,collections
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
ms=json.loads((R/'metrics.json').read_text());success={1,10,16,20,21,28}
for m in ms:
 p=Path(m['path']);rows=[json.loads(x) for x in (p/'trace.jsonl').read_text().splitlines()];pred=[json.loads(x) for x in (p/'prediction_chunks.jsonl').read_text().splitlines()]
 times=np.array([r['simulation_elapsed_s'] for r in rows]);c=[r['contact_status'] for r in rows];pos=np.array([x['object_position_m'] for x in c]);j=np.array([r['joint_state'] for r in rows]);cmd=np.array([r['execution_target'] for r in rows]);dt=np.diff(times,append=times[-1]);height=int(pos[:,2].argmax())
 m['final_box_success']=m['episode'] in success;m['max_pred']=max(max(x['gripper_prediction']) for x in pred)
 m['max_height_time']=float(times[height]);m['max_height_closed']=bool(cmd[height,-1]>-.45);m['max_height_contacts']=[c[height].get(k) for k in ['left_finger_contact_object','right_finger_contact_object']];m['max_height_actual_gripper']=float(j[height,-1])
 m['last10_joint_span']=np.ptp(j[times>times[-1]-10,:6],axis=0).tolist();m['actual_gripper_range']=[float(j[:,-1].min()),float(j[:,-1].max())]
 m['other_contact_finger_seconds']=dict(collections.Counter())
 for name in ['beverage_bottle','apple','peach','mouse']:
  count=np.array([sum(x.get(k)==name for k in ['left_finger_contact_object','right_finger_contact_object']) for x in c]);v=float((count*dt).sum())
  if v:m['other_contact_finger_seconds'][name]=v
 m['events']=[dict(e,actual_gripper=float(j[int(np.argmin(abs(times-e['time']))),-1]),contacts=[c[int(np.argmin(abs(times-e['time'])))].get(k) for k in ['left_finger_contact_object','right_finger_contact_object']]) for e in m['events']]
 print(f"{m['episode']:02} finalOK={m['final_box_success']} C{m['close_events']}O{m['open_events']} predmax={m['max_pred']:.3f} nearErr={m['min_envelope_error_m']*1000:.1f} both={m['both_seconds']:.2f} zmax={m['max_root_z']:.3f}@{m['max_height_time']:.1f} actualjaw={m['actual_gripper_range']} others={m['other_contact_finger_seconds']}")
(R/'metrics.json').write_text(json.dumps(ms,indent=2))
for m in ms:
 if m['episode'] in [2,5,13,22,23,25,26]:print('EVENTS',m['episode'],json.dumps(m['events']))
