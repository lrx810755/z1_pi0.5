import json,threading,subprocess,time,shutil
from pathlib import Path
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger
from z1_isaac_bridge.smolvla_sim_rollout import parse_args
from replay import Replay
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_bottle_collision_consistent_supplement_20261003_raw_v1')
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def main():
 result_path=ROOT/'supplement_collection_result.json'
 if result_path.exists() and not (ROOT/'supplement_initial_24_attempts_result.json').exists():shutil.copy2(result_path,ROOT/'supplement_initial_24_attempts_result.json')
 subprocess.run(['systemd-run','--user','--unit','z1-supplement-physics-linear-approach-20261003','/bin/bash',str(ROOT/'record_physics.sh')],check=True)
 args=parse_args(['--checkpoint',str(DATA),'--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw']);args.training_manifest=None
 rclpy.init();node=Replay(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
 services={name:node.create_client(Trigger,path) for name,path in {
 'generate':'/smolvla/auto_generate_pregrasp','execute':'/smolvla/auto_execute_pregrasp','grasp':'/smolvla/auto_guided_grasp','place':'/smolvla/auto_guided_place','start':'/smolvla/recording/start','save':'/smolvla/recording/save','discard':'/smolvla/recording/discard'}.items()}
 outcomes=[]
 try:
  for seed,grid in [(309454515,[2,0]),(342470491,[3,0])]:
   saved=0
   for attempt in range(1,4):
    if saved>=2:break
    entry={'seed':seed,'grid':grid,'attempt':attempt,'success':False};recording=False
    try:
     node.recorded_scene={'target_grid':grid};node._publish_target('beverage_bottle');node._randomize_scene('beverage_bottle',seed)
     node._call_trigger(services['generate'],420)
     node._call_trigger(node._stop_motion,5)
     node._call_trigger(services['start'],10);recording=True
     for stage,timeout in [('execute',180),('grasp',600),('place',600),('save',30)]:
      entry['stage']=stage;print('VERIFIED_LAYOUT_STAGE',seed,attempt,stage,flush=True);node._call_trigger(services[stage],timeout)
     recording=False;entry['success']=True;saved+=1
    except Exception as exc:
     entry['error']=str(exc)
     if recording:
      try:node._call_trigger(services['discard'],30)
      except Exception:pass
    outcomes.append(entry);save('supplement_linear_approach_attempts.json',outcomes);print('VERIFIED_LAYOUT_ATTEMPT',entry,flush=True)
 finally:
  node._set_active(False);executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
  subprocess.run(['systemctl','--user','stop','z1-supplement-physics-linear-approach-20261003'],check=True)
 episodes=[]
 for folder in sorted(DATA.glob('episode_*')):
  summary=json.loads((folder/'episode_summary.json').read_text());rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
  episodes.append({'path':str(folder),'frames':len(rows),'task_success':summary['task_success'],'seed':summary.get('scene_randomization',{}).get('seed'),'target_grid':summary.get('scene_randomization',{}).get('target_grid'),'middle_frames_removed':0,'physics':'collision_preserving_fixed_approach; guided_unfreeze_before_close','needs_unassisted_replay':True})
 save('supplement_collection_result.json',{'initial_failed_attempts':24,'retry_attempts':outcomes,'groups':[{'grid':grid,'saved':sum(o['success'] and o['grid']==grid for o in outcomes)} for grid in [[1,1],[2,0],[3,0]]],'episodes':episodes,'total_successes':sum(e['task_success'] for e in episodes),'dataset_root':str(DATA),'middle_frame_trimming':False,'physics_history':str(ROOT/'supplement_physics_history.jsonl'),'training_eligible':False,'eligibility_reason':'Pending pure joint action replay; assisted success alone is insufficient.'})
 subprocess.run(['/bin/bash',str(ROOT/'replay_supplement.sh')],check=True)
 subprocess.run(['/usr/bin/python3',str(ROOT/'analyze_four_steps.py')],check=True)
if __name__=='__main__':main()
