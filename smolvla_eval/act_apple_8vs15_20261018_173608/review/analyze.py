import json,collections,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent.parent
out=[]
for steps in [8,15]:
 for p in sorted((R/f'steps{steps}').glob('act_sim_eval_*/episodes/*/result.json')):
  result=json.loads(p.read_text());rows=[json.loads(l) for l in (p.parent/'trace.jsonl').read_text().splitlines()];pred=[json.loads(l) for l in (p.parent/'prediction_chunks.jsonl').read_text().splitlines()]
  times=np.array([r['simulation_elapsed_s'] for r in rows]);cs=[r['contact_status'] for r in rows]
  positions=np.array([c['object_position_m'] for c in cs]);rel=np.array([c['relative_position_gripper_m'] for c in cs]);closed=np.array([r['execution_target'][-1]>-.45 for r in rows]);actual=np.array([r['joint_state'][-1] for r in rows])
  both=np.array([c.get('left_finger_contact_object')=='apple' and c.get('right_finger_contact_object')=='apple' for c in cs]);one=np.array([c.get('left_finger_contact_object')=='apple' or c.get('right_finger_contact_object')=='apple' for c in cs]);verified=np.array([c.get('physical_grasp_ok',False) for c in cs]);env=np.array([c.get('inside_grasp_envelope',False) for c in cs])
  closeids=np.flatnonzero(closed&~np.r_[False,closed[:-1]]);openids=np.flatnonzero(~closed&np.r_[False,closed[:-1]])
  # Physical root height avoids calling a tilted fruit at 7mm "lifted".
  high=positions[:,2]>=.05;bilateral_lift=high&both;boxxy=(abs(positions[:,0]-.32)<=.055)&(abs(positions[:,1]-.14)<=.055)
  # Teacher's loaded apple center envelope, for diagnostics, not a new success gate.
  er=np.stack([np.maximum(.142-rel[:,0],0)+np.maximum(rel[:,0]-.164,0),np.maximum(abs(rel[:,1])-.007,0),np.maximum(-rel[:,2],0)+np.maximum(rel[:,2]-.021,0)],1)
  distance=np.linalg.norm(er,axis=1);nearest=int(np.argmin(distance));near=distance<=.005
  closest_closed=np.flatnonzero(closed);firstclose=int(closeids[0]) if len(closeids) else None
  liftids=np.flatnonzero(bilateral_lift);firstlift=int(liftids[0]) if len(liftids) else None
  highids=np.flatnonzero(high);firsthigh=int(highids[0]) if len(highids) else None
  post_lift_open=[int(i) for i in openids if firsthigh is not None and i>firsthigh]
  open_after=post_lift_open[0] if post_lift_open else None
  high_closed_box=high&closed&boxxy
  candidatepred=sum(x.get('first_close_step_06') is not None for x in pred)
  beyond=sum(x.get('first_close_step_06') is not None and x['first_close_step_06']>=steps for x in pred)
  held_drop=False;dropindex=None
  if firsthigh is not None:
   ids=np.flatnonzero((np.arange(len(rows))>firsthigh)&(positions[:,2]<.02)&closed&~boxxy)
   if len(ids):held_drop=True;dropindex=int(ids[0])
  m=dict(steps=steps,episode=result['episode_number'],seed=result['seed'],success=result['success'],category=result['failure_category'],path=str(p.parent),initial_position=result['target_position_m'],duration=float(times[-1]),close_events=len(closeids),open_events=len(openids),closed_seconds=float(np.diff(times,append=times[-1])[closed].sum()),first_close_s=None if firstclose is None else float(times[firstclose]),first_close_local=None if firstclose is None else rel[firstclose].tolist(),first_close_root=None if firstclose is None else positions[firstclose].tolist(),first_close_envelope_error_m=None if firstclose is None else float(distance[firstclose]),first_close_displacement_m=None if firstclose is None else float(np.linalg.norm(positions[firstclose,:2]-positions[0,:2])),min_envelope_error_m=float(distance[nearest]),closest_time=float(times[nearest]),closest_local=rel[nearest].tolist(),near_seconds=float(np.diff(times,append=times[-1])[near].sum()),near_closed_seconds=float(np.diff(times,append=times[-1])[near&closed].sum()),ever_both=bool(both.any()),both_seconds=float(np.diff(times,append=times[-1])[both].sum()),one_seconds=float(np.diff(times,append=times[-1])[one].sum()),ever_verified=bool(verified.any()),max_root_z=float(positions[:,2].max()),max_xy_displacement=float(np.linalg.norm(positions[:,:2]-positions[0,:2],axis=1).max()),ever_bilateral_lift=bool(bilateral_lift.any()),first_high_s=None if firsthigh is None else float(times[firsthigh]),box_high_closed_seconds=float(np.diff(times,append=times[-1])[high_closed_box].sum()),first_open_after_high_s=None if open_after is None else float(times[open_after]),first_open_after_high_position=None if open_after is None else positions[open_after].tolist(),first_open_after_high_inside_box=None if open_after is None else bool(boxxy[open_after]),drop_while_closed=held_drop,drop_s=None if dropindex is None else float(times[dropindex]),final_root=positions[-1].tolist(),final_actual_gripper=float(actual[-1]),final_closed=bool(closed[-1]),final_inside_box_xy=bool(boxxy[-1]),prediction_chunks=len(pred),chunks_predict_close=candidatepred,chunks_close_beyond_execution=beyond,events=[dict(event='close' if closed[i] else 'open',time=float(times[i]),local=rel[i].tolist(),root=positions[i].tolist(),inside_box=bool(boxxy[i]),near_error=float(distance[i])) for i in sorted([*closeids,*openids])])
  out.append(m)
(R/'review'/'metrics.json').write_text(json.dumps(out,indent=2))
for steps in [8,15]:
 print('\nGROUP',steps)
 for m in sorted([x for x in out if x['steps']==steps],key=lambda m:m['episode']):
  print(f"{m['episode']:02d} {m['category']:27s} C{m['close_events']:2d} O{m['open_events']:2d} first={str(None if m['first_close_s'] is None else round(m['first_close_s'],1)):>4s} err={str(None if m['first_close_envelope_error_m'] is None else round(m['first_close_envelope_error_m']*1000)):>4s}mm near={m['near_seconds']:.1f}s both={m['both_seconds']:.1f}s z={m['max_root_z']:.3f} box={m['box_high_closed_seconds']:.1f}s dropClosed={m['drop_while_closed']} openBox={m['first_open_after_high_inside_box']} future={m['chunks_close_beyond_execution']}/{m['chunks_predict_close']}")
