import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
summary=[]
for result in sorted(ROOT.rglob('result.json')):
 r=json.loads(result.read_text());folder=result.parent
 chunks=[json.loads(x) for x in (folder/'prediction_chunks.jsonl').read_text().splitlines()]
 trace=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
 sync=[json.loads(x) for x in (folder/'input_sync.jsonl').read_text().splitlines()]
 base=chunks[0]['sim_stamp_s'] if chunks else 0
 query_rows=[];outside=0;postponed=[]
 for i,x in enumerate(chunks):
  k=x['first_close_step_06'];query_rows.append(dict(query=i,sim_time_relative_first_query_s=x['sim_stamp_s']-base,first_close_action_1based=None if k is None else k+1,within_execution_window=x['close_in_execution_window'],first_gripper_prediction=x['gripper_prediction'][0],gripper_prediction=x['gripper_prediction']))
  if k is not None and not x['close_in_execution_window']:
   outside+=1
   if i+1<len(chunks):
    nxt=chunks[i+1];nk=nxt['first_close_step_06']
    # Compare nominal planned absolute close time with the next plan.
    delta=None if nk is None else (nxt['sim_stamp_s']+nk*.1)-(x['sim_stamp_s']+k*.1)
    postponed.append(dict(query=i,first_close_action_1based=k+1,next_query=i+1,next_close_action_1based=None if nk is None else nk+1,nominal_close_time_shift_s=delta,next_plan_has_no_close=nk is None))
 closed=[x for x in trace if x['execution_target'][-1]==0]
 predicted=[x for x in chunks if x['first_close_step_06'] is not None]
 group=ROOT/f"steps{r['execute_steps']}";name=folder.name
 output=group/f'{name}_queries.json';output.write_text(json.dumps(query_rows,indent=2))
 summary.append(dict(steps=r['execute_steps'],seed=r['seed'],success=r['success'],failure=r['failure_category'],simulation_duration_s=r['simulation_duration_s'],initial_target_position=r['target_position_m'],queries=len(chunks),queries_with_close=len(predicted),queries_close_beyond_window=outside,first_actual_close_s=closed[0]['simulation_elapsed_s'] if closed else None,actual_close_target_rows=len(closed),postponements=postponed,query_file=str(output),result=str(result)))
(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
