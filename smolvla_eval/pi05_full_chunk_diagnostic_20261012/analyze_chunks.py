from pathlib import Path
import json,math
P=Path(__file__).resolve().parent
OLD=Path('/home/lp531/smolvla_eval/pi05_paired_comparison_20261011')
summary=[]
for f in sorted(P.glob('*/**/predictions.jsonl')):
 rows=[json.loads(x) for x in f.read_text().splitlines()];rf=f.parent/'result.json';trace=[json.loads(x) for x in (f.parent/'trace.jsonl').read_text().splitlines()]
 if not rows:continue
 result=json.loads(rf.read_text()) if rf.exists() else None
 name=f.relative_to(P).parts[0];seed=int(f.parent.name.split('_')[-1]);prefix=[r for r in rows if r['executed_prefix_close_indices']];tailonly=[r for r in rows if not r['executed_prefix_close_indices'] and r['discarded_tail_close_indices']];allopen=[r for r in rows if not r['executed_prefix_close_indices'] and not r['discarded_tail_close_indices']]
 # Tie full chunks to actually executed action rows by chunk start resets.
 actionrows=[r for r in trace if r['execution_substep']==0];chunk=-1;matched=0;mismatches=[]
 for t in actionrows:
  if t['policy_action_index']==0:chunk+=1
  i=t['policy_action_index']
  if chunk>=len(rows) or any(abs(x-y)>1e-7 for x,y in zip(t['policy_action'],rows[chunk]['actions'][i])):mismatches.append(dict(chunk=chunk,index=i))
  else:matched+=1
 sync=[json.loads(x) for x in (f.parent/'input_sync.jsonl').read_text().splitlines()]
 initial={}
 if result:
  original=json.loads(next((OLD/name).glob(f'*/episodes/*_seed_{seed}/result.json')).read_text())
  bp=original['initial_scene_randomization']['object_positions_m'];np_=result['initial_scene_randomization']['object_positions_m'];initial=dict(max_object_position_difference_m=max(math.dist(bp[k],np_[k]) for k in bp),max_joint_difference_rad=max(abs(x-y) for x,y in zip(original['initial_joint_state'],result['initial_joint_state'])))
 near=[r for r in rows if r.get('contact_status',{}).get('inside_grasp_envelope')]
 def brief(r):
  return dict(prediction=r['prediction'],first_close_index=r['first_close_index'],gripper=r['gripper_raw'],relative=r['contact_status'].get('relative_position_gripper_m'),distance=r['contact_status'].get('gripper_object_distance_m'),phase=r['contact_status'].get('phase'),input_stamp=r['input_sync'].get('image_stamps_s'))
 transitions=[]
 for r,next_r in zip(rows,rows[1:]):
  if not r['executed_prefix_close_indices'] and r['discarded_tail_close_indices']:
   transitions.append(dict(prediction=r['prediction'],first_close_index=r['first_close_index'],next_first_close_index=next_r['first_close_index']))
 d=dict(model=name,seed=seed,complete=result is not None,success=result.get('success') if result else None,failure=result.get('failure_category') if result else None,predictions=len(rows),chunk_lengths=sorted(set(r['chunk_length'] for r in rows)),prefix_close_chunks=len(prefix),tail_only_close_chunks=len(tailonly),all_open_chunks=len(allopen),full_chunk_gripper_min=min(min(r['gripper_raw']) for r in rows),full_chunk_gripper_max=max(max(r['gripper_raw']) for r in rows),tail_only_events=[brief(r) for r in tailonly],first_prefix_close=brief(prefix[0]) if prefix else None,tail_only_followups=transitions,inside_loaded_envelope_prediction_count=len(near),inside_loaded_allopen_count=sum(r in allopen for r in near),executed_actions_checked=matched,action_mismatches=mismatches,input_sync_count=len(sync),input_ids_match=[r['prediction'] for r in rows]==[r['prediction'] for r in sync],initial_state_vs_prior=initial,path=str(f.parent))
 summary.append(d)
report=dict(expected_episodes=3,complete_episodes=sum(x['complete'] for x in summary),episodes=summary)
(P/'chunk_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for d in summary:print({k:d[k] for k in ['model','seed','complete','success','predictions','chunk_lengths','prefix_close_chunks','tail_only_close_chunks','all_open_chunks','full_chunk_gripper_max','executed_actions_checked']})
