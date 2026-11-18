from check import *
import csv,hashlib
REF=np.array([.12083045,-.0028452,.0194797])
LIMITS=np.array([[float(J[f'joint{i}'].find('limit').attrib[k]) for k in ['lower','upper']] for i in range(1,7)])
def stats(a):
 a=np.asarray(a);return {k:float(v) for k,v in zip(['median','p95','max'],np.percentile(a,[50,95,100]))} if len(a) else None
def future(t,q,lag):return np.column_stack([np.interp(t+lag,t,q[:,j]) for j in range(6)])
rows=[];allmetrics={k:[] for k in ['lag_command_mm','endpoint_target_mm','endpoint_command_mm','fk_tf_mm','sensor_age_ms','skew_ms','inference_sim_ms','inference_wall_ms']}
for e in json.loads((R/'结果与失败分类.json').read_text())['episodes']:
 folder=pathlib.Path(e['source_result']).parent
 trace=[json.loads(l) for l in (folder/'trace.jsonl').read_text().splitlines()]
 sync=[json.loads(l) for l in (folder/'input_sync.jsonl').read_text().splitlines()]
 preds=[json.loads(l) for l in (folder/'prediction_chunks.jsonl').read_text().splitlines()]
 a={k:np.array([z[k] for z in trace]) for k in ['joint_state','policy_action','command','execution_target']}
 q=a['joint_state'][:,:6];target=a['policy_action'][:,:6];cmd=a['command'][:,:6];t=np.array([z['simulation_elapsed_s'] for z in trace]);n=len(t)
 # Poses for all three stages use the same cup grasp reference offset; no base fitting.
 actual,rot=fk(q,REF);goal,_=fk(target,REF);sent,_=fk(cmd,REF)
 original,_=fk(q);tf=np.array([z['contact_status']['gripper_world_position_m'] for z in trace]);fkerr=np.linalg.norm(original-tf,axis=1)
 obj=np.array([z['contact_status']['object_position_m'] for z in trace]);oq=np.array([z['contact_status']['object_orientation_xyzw'] for z in trace]);x,y,z,w=oq.T
 center=obj+np.column_stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)))*.0305
 initial=np.array(e['initial_cup_position_m']);moved=np.linalg.norm(obj-initial,axis=1)>.03
 firstmove=int(np.flatnonzero(moved)[0]) if moved.any() else n
 pre=np.arange(n)<firstmove;dist=np.linalg.norm(actual-center,axis=1);near=pre&(dist<.12)
 endpoint=np.array([z['execution_substep']==7 for z in trace]);mask=near&endpoint
 if mask.sum()<5:mask=pre&endpoint
 # Command timestamp precedes the responding joint state. Search physical latency using moving commands.
 moving=np.r_[False,np.max(abs(np.diff(cmd,axis=0)),axis=1)>.0002]&pre
 valid=moving&(t<t[-1]-.2)
 lags=np.arange(0,.151,.008333333333)
 fits=[]
 for lag in lags:
  futureq=future(t,q,lag);fits.append(float(np.sqrt(np.mean((futureq[valid]-cmd[valid])**2))))
 bestlag=float(lags[int(np.argmin(fits))]);follow=future(t,q,bestlag);followp,_=fk(follow,REF)
 future_cmd_error=np.linalg.norm(followp-sent,axis=1)*1000
 # Last tick residual is measured before that tick's command; report separately from future tracking.
 end_target=np.linalg.norm(actual-goal,axis=1)*1000;end_cmd=np.linalg.norm(actual-sent,axis=1)*1000
 clipping=np.any((target<LIMITS[:,0])|(target>LIMITS[:,1]),axis=1)
 starts=np.array([z['execution_substep']==0 for z in trace]);limitstep=np.max(abs(np.clip(target,LIMITS[:,0],LIMITS[:,1])-q),axis=1)>.10001
 ages=np.array([s['oldest_sensor_age_s'] for s in sync])*1000;skew=np.array([s['sensor_skew_s'] for s in sync])*1000
 predsim=np.array([(p['sim_stamp_s']-s['latest_state_time_s'])*1000 for p,s in zip(preds,sync)])
 # Approximate forward latency: predict() records its sim_stamp before forward; next executed row is after forward.
 firstrows=[i for i,z in enumerate(trace) if z['execution_substep']==0 and z['policy_action_index']==0]
 relstart=float(np.median([trace[i]['contact_status']['gripper_tf_stamp_s']-t[i] for i in firstrows]))
 # The start offset is uncertain because contact TF can be older; retain only exact sync statistics.
 nearest=int(np.argmin(np.where(pre,dist,np.inf)))
 row=dict(episode=e['episode'],success=e['success'],category=e['observed_category'],trace_rows=n,pre_displacement_rows=int(pre.sum()),near_reference_rows=int(near.sum()),best_command_lag_ms=bestlag*1000,command_follow_rmse_deg=min(fits)*180/np.pi,command_follow_near_mm=stats(future_cmd_error[mask]),target_residual_near_end_mm=stats(end_target[mask]),command_residual_near_end_mm=stats(end_cmd[mask]),nearest_actual_reference_mm=float(dist[nearest]*1000),at_nearest_goal_reference_mm=float(np.linalg.norm(goal[nearest]-center[nearest])*1000),at_nearest_command_reference_mm=float(np.linalg.norm(sent[nearest]-center[nearest])*1000),at_nearest_goal_actual_mm=float(end_target[nearest]),at_nearest_actual_minus_cup_mm=((actual[nearest]-center[nearest])*1000).tolist(),joint_limit_clipped_action_start_fraction=float(clipping[starts].mean()),step_limited_action_start_fraction=float(limitstep[starts].mean()),fk_vs_recorded_tf_mm=stats(fkerr*1000),sensor_age_ms=stats(ages),sensor_skew_ms=stats(skew),policy_paused_rows=sum(bool(z['policy_action_paused']) for z in trace),first_cup_3cm_displacement_s=float(t[firstmove]) if firstmove<n else None)
 rows.append(row)
 for k,v in [('lag_command_mm',future_cmd_error[mask]),('endpoint_target_mm',end_target[mask]),('endpoint_command_mm',end_cmd[mask]),('fk_tf_mm',fkerr*1000),('sensor_age_ms',ages),('skew_ms',skew)]:allmetrics[k].extend(v.tolist())
 print(e['episode'],'lag',round(bestlag*1000,1),'follow_mm',round(row['command_follow_near_mm']['p95'],2),'target_gap_mm',round(row['target_residual_near_end_mm']['p95'],2),'closest',round(dist[nearest]*1000,1),flush=True)
summary=dict(total=30,reference_point_local_m=REF.tolist(),reference_source='cup calibrated_grasp_reference in current grasp_physics.json; geometry-derived and marked pending validation, used as an analysis reference, not an exact measured finger midpoint',pre_displacement_definition='stop analysis at first cup root displacement above3cm; near means reference-to-cup-center distance below12cm; last execution_substep7',latency_method='offline best fit over0..150ms of issued command to future joint state; moving commands before3cm cup displacement; observational rather than controlled step response',metrics={k:stats(v) for k,v in allmetrics.items() if v},best_command_lag_ms=stats([x['best_command_lag_ms'] for x in rows]),command_follow_rmse_deg=stats([x['command_follow_rmse_deg'] for x in rows]),episodes_never_nearer_than5cm=sum(x['nearest_actual_reference_mm']>50 for x in rows),episodes_policy_pause=sum(x['policy_paused_rows']>0 for x in rows),episodes=rows)
(R/'execution_audit'/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with (R/'execution_audit'/'episodes.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k!='episodes'},ensure_ascii=False,indent=2))
