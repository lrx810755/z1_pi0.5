import json,collections
from pathlib import Path
import numpy as np
source=Path('/home/lp531/smolvla_eval/act_15steps_hold1s_30_trials_20261014/review/trigger_analysis.py')
ns={'__file__':str(source)};exec(compile(source.read_text().split('records=[]')[0],str(source),'exec'),ns)
angles=ns['angles'];span=ns['span'];env=ns['env'];limit=ns['limit']
W=Path(__file__).resolve().parent;ROOT=W.parent
records=[]
for steps in [15,8]:
 for p in sorted((ROOT/f'steps{steps}').glob('act_sim_eval_*/episodes/*/result.json')):
  r=json.loads(p.read_text());rows=[json.loads(x) for x in (p.parent/'trace.jsonl').read_text().splitlines()];queries=[json.loads(x) for x in (p.parent/'prediction_chunks.jsonl').read_text().splitlines()]
  valid=[x for x in rows if x.get('contact_status',{}).get('relative_position_gripper_m') is not None]
  t=np.array([x['simulation_elapsed_s'] for x in valid]);v=np.array([x['contact_status']['relative_position_gripper_m'] for x in valid]);a=angles(np.array([x['joint_state'][:6] for x in valid]));cmd=np.array([x['execution_target'][-1] for x in valid]);pred=np.array([x['policy_action'][-1] for x in valid]);pre=np.cumsum(cmd==0)==0
  err=np.column_stack([np.maximum(np.maximum(env['x_min_m']-v[:,0],v[:,0]-env['x_max_m']),0),np.maximum(abs(v[:,1])-env['y_abs_max_m'],0),np.maximum(np.maximum(env['z_min_m']-v[:,2],v[:,2]-env['z_max_m']),0)])
  position=err.max(axis=1)<1e-9;candidate=position&(a<=limit)&pre
  before=np.flatnonzero(pre);best=before[np.argmin(np.linalg.norm(err[before],axis=1))] if len(before) else 0
  close=np.flatnonzero(cmd==0);switch=[];last=-.9
  for row in rows:
   z=row['execution_target'][-1]
   if z!=last:switch.append(dict(time_s=row['simulation_elapsed_s'],target=z));last=z
  final=r['final_status'];holding=final.get('holding_observed',False);lifted=any(x['lifted'] for x in rows);verified=any(x['physical_grasp_ok'] for x in rows)
  rec=dict(steps=steps,episode=r['episode_number'],seed=r['seed'],failure=r['failure_category'],success=r['success'],ever_close=bool(len(close)),ever_actual_closed=any(x['joint_state'][-1]>=-.75 for x in rows),ever_lifted=lifted,ever_verified=verified,final_holding=holding,final_bilateral=final.get('bilateral_contact'),final_released=final.get('released'),final_position=final.get('object_position_m'),target_position=r['target_position_m'],ever_position_before_close=bool((position&pre).any()),candidate_before_close=bool(candidate.any()),candidate_max_duration_s=span(candidate,t),candidate_prediction_max=float(pred[candidate].max()) if candidate.any() else None,full_chunk_max=max(max(q['gripper_prediction']) for q in queries),queries=len(queries),queries_with_close=sum(q['first_close_step_06'] is not None for q in queries),queries_close_outside_window=sum(q['first_close_step_06'] is not None and q['first_close_step_06']>=steps for q in queries),hold_blocked_rows=sum(x.get('gripper_close_hold_blocked_open',False) for x in rows),close_cycles=sum(x['target']==0 for x in switch),switches=switch,closest_before_close=dict(time_s=float(t[best]),relative_position_mm=(v[best]*1000).tolist(),outside_mm=(err[best]*1000).tolist(),tilt_deg=float(a[best]),prediction=float(pred[best])),first_close=dict(time_s=float(t[close[0]]),position_mm=(v[close[0]]*1000).tolist()) if len(close) else None,result=str(p))
  records.append(rec)
summary={}
for steps in [15,8]:
 rr=[r for r in records if r['steps']==steps];no=[r for r in rr if not r['ever_close']]
 summary[str(steps)]=dict(episodes=len(rr),successes=sum(r['success'] for r in rr),categories=dict(collections.Counter(r['failure'] for r in rr)),never_close=len(no),never_close_categories=dict(collections.Counter(r['failure'] for r in no)),never_close_chunk_below04=sum(r['full_chunk_max']<.4 for r in no),never_close_candidate=sum(r['candidate_before_close'] for r in no),never_close_position=sum(r['ever_position_before_close'] for r in no),ever_close=sum(r['ever_close'] for r in rr),ever_lifted=sum(r['ever_lifted'] for r in rr),ever_verified=sum(r['ever_verified'] for r in rr),final_holding=[r['episode'] for r in rr if r['final_holding']],repeated_close=[r['episode'] for r in rr if r['close_cycles']>1],ever_candidate_before_close=sum(r['candidate_before_close'] for r in rr),close_predictions_only_outside_window_ever=[r['episode'] for r in rr if r['queries_close_outside_window']],hold_blocked=[r['episode'] for r in rr if r['hold_blocked_rows']])
pairs=[]
for x in [r for r in records if r['steps']==15]:
 y=next(r for r in records if r['steps']==8 and r['seed']==x['seed']);pairs.append(dict(episode=x['episode'],seed=x['seed'],same_category=x['failure']==y['failure'],steps15_close=x['ever_close'],steps8_close=y['ever_close'],steps15_candidate=x['candidate_before_close'],steps8_candidate=y['candidate_before_close']))
report=dict(summary=summary,pairs=pairs,episodes=records,preclose_envelope=env,orientation_limit_deg=limit,limitations=['Configuration position envelope is only a reference, not empirical demonstrated graspability.','Tilt compared to held-out demonstration reference; no bottle tilt per frame.','Executed actions originate from earlier observations; logs cannot prove visual causal trigger failure.','Lifted is not verified grasp.'])
(W/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2));print('same_paired_category',sum(x['same_category'] for x in pairs))
for r in records:
 if r['ever_close'] or r['candidate_before_close']:print('DETAIL',json.dumps(r,ensure_ascii=False))
