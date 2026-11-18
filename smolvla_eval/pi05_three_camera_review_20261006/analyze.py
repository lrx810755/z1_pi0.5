from pathlib import Path
import json, numpy as np
ROOT=Path('/home/lp531/smolvla_eval/pi05_bottle_three_camera_best')
out=[]
for p in sorted(ROOT.rglob('result.json')):
 d=json.loads(p.read_text());rows=[json.loads(l) for l in (p.parent/'trace.jsonl').read_text().splitlines()]
 sync=[json.loads(l) for l in (p.parent/'input_sync.jsonl').read_text().splitlines()]
 pred=np.array([r['policy_action'] for r in rows]);cmd=np.array([r['command'] for r in rows]);q=np.array([r['joint_state'] for r in rows]);times=np.array([r['simulation_elapsed_s'] for r in rows]);loc=np.array([r['contact_status']['relative_position_gripper_m'] for r in rows])
 inside=(loc[:,0]>=.110)&(loc[:,0]<=.154)&(abs(loc[:,1])<=.04)&(loc[:,2]>=.006)&(loc[:,2]<=.03)
 contact=np.array([r['contact_status'].get('left_finger_contact_object')=='beverage_bottle' or r['contact_status'].get('right_finger_contact_object')=='beverage_bottle' for r in rows])
 first=lambda mask:float(times[np.flatnonzero(mask)[0]]) if mask.any() else None
 close=np.flatnonzero(cmd[:,-1]>=-.75)
 ends=np.array([r['execution_substep']==5 for r in rows])
 item=dict(run=p.parents[2].name,reason=d['stop_reason'] or d['failure_category'],sim_s=d['simulation_duration_s'],pred_gripper_range=[float(pred[:,-1].min()),float(pred[:,-1].max())],cmd_gripper_max=float(cmd[:,-1].max()),actual_gripper_max=float(q[:,-1].max()),first_close_s=first(cmd[:,-1]>=-.75),first_target_contact_s=first(contact),inside_envelope_rows=int(inside.sum()),closest_local_to_center=loc[np.argmin(np.linalg.norm(loc-[.132,0,.018],axis=1))].tolist(),first_close_local=loc[close[0]].tolist() if len(close) else None,final_local=d['final_status']['relative_position_gripper_m'],max_sensor_skew_ms=max([r.get('sensor_skew_s',0) for r in sync])*1000,camera_sets=sorted(set(tuple(sorted(r.get('image_stamps_s',{}))) for r in sync)),frozen_contact_rows=sum(r['command_target_frozen'] and c for r,c in zip(rows,contact)),max_pred_cmd_gripper_endpoint=float(np.max(np.abs(pred[ends,-1]-cmd[ends,-1]))) if ends.any() else None,terminal_arm_tracking_abs=(np.abs(cmd[-1,:6]-q[-1,:6])).tolist())
 out.append(item)
raw=Path('/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1');train=[]
for p in sorted(raw.glob('episode_[0-9][0-9][0-9][0-9][0-9][0-9]')):
 rows=[json.loads(l) for l in (p/'frames.jsonl').read_text().splitlines()];a=np.array([r['action'] for r in rows]);ix=np.flatnonzero(a[:,-1]>=-.75)
 train.append(dict(episode=p.name,first_close_s=rows[ix[0]]['timestamp'] if len(ix) else None,max_gripper=float(a[:,-1].max()),closed_fraction=float(np.mean(a[:,-1]>=-.75))))
summary={'evals':out,'train':{'episodes':len(train),'episodes_with_close':sum(x['first_close_s'] is not None for x in train),'first_close_s_quantiles':np.quantile([x['first_close_s'] for x in train],[0,.5,1]).tolist(),'max_gripper_quantiles':np.quantile([x['max_gripper'] for x in train],[0,.5,1]).tolist(),'closed_fraction_median':float(np.median([x['closed_fraction'] for x in train]))}}
Path(__file__).with_name('analysis.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps(summary,indent=2,default=lambda x:x.item()))
