import collections,json
from pathlib import Path
R=Path(__file__).resolve().parent

def summarize():
 plan=json.loads((R/'selected_scenes.json').read_text());expected={s['seed']:s for s in plan['scenes']};files=list((R/'rollouts').glob('act_sim_eval_*/episodes/*/result.json'))
 if len(files)!=10:raise RuntimeError(f'Expected ten completed results, found {len(files)}')
 reproduced=[json.loads(l) for l in (R/'scene_reproduction.jsonl').read_text().splitlines()]
 assert len(reproduced)==10 and {r['seed'] for r in reproduced}==set(expected)
 episodes=[]
 for path in sorted(files,key=lambda p:json.loads(p.read_text())['episode_number']):
  d=json.loads(path.read_text())
  if d.get('seed') not in expected or d.get('failure_category')=='episode_setup_failed' or d.get('error_detail') or d.get('cleanup_error'):raise RuntimeError(f'Invalid diagnostic episode: {path}')
  rows=[json.loads(l) for l in (path.parent/'trace.jsonl').open()];s=d['final_status'];xyz=s['object_position_m'];close=False;events=[]
  for t in rows:
   now=t['execution_target'][-1]>-.75
   if now!=close:events.append(dict(kind='close' if now else 'open',simulation_s=t['simulation_elapsed_s'],relative=t['contact_status'].get('relative_position_gripper_m'),was_grasped=t['contact_status'].get('was_grasped')))
   close=now
  success=abs(xyz[0]-.32)<=.055 and abs(xyz[1]-.14)<=.055 and 0<=xyz[2]<=.12 and s.get('released') is True and not s.get('holding_observed')
  videos={cam:str(path.parent/'videos'/cam) for cam in ['global_third_person.mp4','global_right.mp4','wrist.mp4']}
  assert all(Path(p).stat().st_size>0 for p in videos.values())
  episodes.append(dict(order=d['episode_number'],training_episode=expected[d['seed']]['training_episode'],grid=expected[d['seed']]['grid'],seed=d['seed'],success=success,failure_category=d['failure_category'],ever_grasp_envelope=any(t['contact_status'].get('inside_grasp_envelope') for t in rows),ever_physical_grasp=any(t['contact_status'].get('physical_grasp_ok') for t in rows),closed_once=any(e['kind']=='close' for e in events),events=events,final_position_m=xyz,videos=videos,source_result=str(path)))
 assert len({e['seed'] for e in episodes})==10
 result=dict(completed=True,diagnostic_only=True,total=10,successes=sum(e['success'] for e in episodes),ever_grasped=sum(e['ever_physical_grasp'] for e in episodes),never_closed=sum(not e['closed_once'] for e in episodes),never_entered_grasp_envelope=sum(not e['ever_grasp_envelope'] for e in episodes),failure_categories=dict(collections.Counter(e['failure_category'] for e in episodes)),episodes=episodes)
 result['success_rate']=result['successes']/10
 (R/'diagnostic_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
if __name__=='__main__':
 d=summarize();print(json.dumps({k:v for k,v in d.items() if k!='episodes'},ensure_ascii=False,indent=2))
