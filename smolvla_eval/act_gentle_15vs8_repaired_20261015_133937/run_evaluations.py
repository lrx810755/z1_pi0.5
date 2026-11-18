import subprocess,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ENTRY='/home/lp531/ros2_ws/src/z1_ros2/scripts/run_smolvla_left_wrist_batch4_eval.sh'
def save(name,value):
 p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
def run(steps,seeds=None):
 cmd=['bash',ENTRY,'--model','10','--execute-steps',str(steps),'--rate','10','--execution-rate','60','--gripper-min-close-hold-s','1','--episodes-per-object','30','--output',str(ROOT/f'steps{steps}')]
 if seeds is not None:cmd+=['--eval-seeds',*[str(s) for s in seeds]]
 save(f'command_steps{steps}.json',cmd)
 existing=list((ROOT/f'steps{steps}').glob('act_sim_eval_*/episodes/*/result.json'))
 existing_reports=list((ROOT/f'steps{steps}').glob('act_sim_eval_*/report.json'))
 if len(existing)==30 and existing_reports:
  code=0  # Verify and reuse the completed group instead of replaying it.
 else:
  with (ROOT/f'steps{steps}.log').open('a',buffering=1) as f:
   p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
   save('status.json',dict(status='running',execute_steps=steps,pid=p.pid,target_episodes=30,next_steps=8 if steps==15 else None,seed_mode='repeat previous 30 random seeds with repaired evaluator',started=datetime.datetime.now().isoformat()))
   code=p.wait()
 results=[(p,json.loads(p.read_text())) for p in (ROOT/f'steps{steps}').glob('act_sim_eval_*/episodes/*/result.json')]
 results.sort(key=lambda x:x[1]['episode_number'])
 reports=list((ROOT/f'steps{steps}').glob('act_sim_eval_*/report.json'))
 save(f'finished_steps{steps}.json',dict(returncode=code,episodes=len(results),reports=[str(p) for p in reports],finished=datetime.datetime.now().isoformat()))
 if code or len(results)!=30 or not reports:raise RuntimeError(f'steps{steps} incomplete: exit {code}, {len(results)} episodes')
 for p,r in results:
  if not (p.parent/'trace.jsonl').stat().st_size or not (p.parent/'prediction_chunks.jsonl').stat().st_size:raise RuntimeError(f'Missing trace at {p}')
 tf_checks=[]
 for p,r in results:
  rows=[json.loads(line) for line in (p.parent/'trace.jsonl').read_text().splitlines()]
  deltas=[]
  for row in rows:
   c=row.get('contact_status',{})
   if c.get('gripper_tf_stamp_s') is not None and c.get('object_pose_stamp_s') is not None:
    deltas.append(abs(c['gripper_tf_stamp_s']-c['object_pose_stamp_s']))
  if not deltas or max(deltas)>.100001:raise RuntimeError(f'Invalid evaluator TF timestamps at {p}')
  for name in ['wrist.mp4','global_third_person.mp4','global_right.mp4']:
   video=p.parent/'videos'/name
   if not video.is_file() or video.stat().st_size==0:raise RuntimeError(f'Missing camera video {video}')
  tf_checks.append(dict(episode=r['episode_number'],max_tf_object_delta_s=max(deltas),trace_rows=len(rows),stamped_rows=len(deltas)))
 save(f'tf_verified_steps{steps}.json',tf_checks)
 actual=[r['seed'] for _,r in results]
 if len(set(actual))!=30:raise RuntimeError('Duplicate random seeds')
 if seeds is not None and actual!=seeds:raise RuntimeError('Paired seeds mismatch')
 return results,actual
try:
 seeds=json.loads((ROOT/'random_seeds.json').read_text())
 a,_=run(15,seeds)
 b,_=run(8,seeds)
 pairs=[]
 for (_,x),(_,y) in zip(a,b):
  if max(abs(v-w) for v,w in zip(x['target_position_m'],y['target_position_m']))>.0001:raise RuntimeError(f'Paired scene position mismatch seed {x["seed"]}')
  pairs.append(dict(seed=x['seed'],steps15_success=x['success'],steps8_success=y['success'],steps15_failure=x['failure_category'],steps8_failure=y['failure_category']))
 save('completed.json',dict(completed=True,episodes_per_group=30,total_episodes=60,paired_seeds=True,position_agreement_within_m=.0001,finished=datetime.datetime.now().isoformat(),pairs=pairs))
 save('status.json',dict(status='complete',total_episodes=60))
except Exception as e:
 save('status.json',dict(status='failed',error=str(e),time=datetime.datetime.now().isoformat()));raise
