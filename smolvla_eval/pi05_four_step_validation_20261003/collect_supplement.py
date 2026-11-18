import json,subprocess,time
from pathlib import Path
from z1_isaac_bridge.continuous_training import JsonApiClient,ContinuousTrainingRunner
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_bottle_collision_consistent_supplement_20261003_raw_v1')
def save(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def main():
 while not all((ROOT/'corrected_comparison'/arm/'completed.json').exists() for arm in ['baseline','uniform','weighted']):
  state=subprocess.check_output(['systemctl','--user','show','pi05-corrected-comparison-20261003','-p','ActiveState','--value'],text=True).strip()
  if state!='active':raise RuntimeError('Corrected comparison stopped before all arms completed')
  time.sleep(2)
 subprocess.run(['/bin/bash',str(ROOT/'replay_legacy_diagnostic.sh')],check=True)
 subprocess.run(['systemctl','--user','restart','z1-collision-preserving-scene-20261003','z1-collision-consistent-assistant-20261003','z1-collision-aware-evaluator-20261003'],check=True)
 subprocess.run(['systemd-run','--user','--unit','z1-supplement-physics-record-20261003','/bin/bash',str(ROOT/'record_physics.sh')],check=True)
 time.sleep(8)
 subprocess.run(['/bin/bash',str(ROOT/'replay_geometry_diagnostic.sh')],check=True)
 api=JsonApiClient('http://127.0.0.1:8767')
 groups=[]
 for grid in [(1,1),(2,0),(3,0)]:
  runner=ContinuousTrainingRunner(api,'beverage_bottle','full',2,target_grid=grid,max_attempts=8,max_failures=8)
  try:
   code=runner.run();entry={'grid':grid,'code':code,'saved':runner.successes,'attempts':runner.attempts,'failures':runner.failures}
  except Exception as exc:
   entry={'grid':grid,'saved':runner.successes,'attempts':runner.attempts,'failures':runner.failures,'error':str(exc)}
  groups.append(entry);save('supplement_collection_progress.json',groups)
  print('SUPPLEMENT_GROUP',entry,flush=True)
 episodes=[]
 for folder in sorted(DATA.glob('episode_*')):
  summary_path=folder/'episode_summary.json'
  if not summary_path.exists():continue
  summary=json.loads(summary_path.read_text()); rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
  episodes.append({'path':str(folder),'frames':len(rows),'task_success':summary.get('task_success'),'seed':summary.get('randomization_seed'),'target_grid':summary.get('scene_randomization',{}).get('target_grid'),'middle_frames_removed':0,'physics':'fixed_target_keeps_collisions; dynamic_at_guided_acceptance','needs_unassisted_replay':True})
 save('supplement_collection_result.json',{'groups':groups,'episodes':episodes,'total_successes':sum(e['task_success'] is True for e in episodes),'dataset_root':str(DATA),'middle_frame_trimming':False,'physics_history':str(ROOT/'supplement_physics_history.jsonl'),'training_eligible':False,'eligibility_reason':'New demonstrations must pass pure joint action replay before merging into training.'})
 subprocess.run(['systemctl','--user','stop','z1-supplement-physics-record-20261003'],check=True)
 subprocess.run(['/bin/bash',str(ROOT/'replay_supplement.sh')],check=True)
 subprocess.run(['/usr/bin/python3',str(ROOT/'analyze_four_steps.py')],check=True)
if __name__=='__main__':main()
