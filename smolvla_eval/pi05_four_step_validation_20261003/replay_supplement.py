import json,threading
from pathlib import Path
import rclpy
from rclpy.executors import MultiThreadedExecutor
from z1_isaac_bridge.smolvla_sim_rollout import parse_args
from replay import Replay
ROOT=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_bottle_collision_consistent_supplement_20261003_raw_v1')
def main():
 manifest=ROOT/'supplement_collection_result.json'
 record=json.loads(manifest.read_text());folders=[Path(e['path']) for e in record['episodes'] if e['task_success']]
 if not folders:
  record.update(replay_completed=True,replay_eligible_episodes=[],eligibility_reason='No successful collision-consistent demonstrations were collected.')
  manifest.write_text(json.dumps(record,indent=2,ensure_ascii=False));return
 args=parse_args(['--checkpoint',str(DATA),'--objects','beverage_bottle','--episodes-per-object',str(len(folders)),'--execute-steps','1','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--freeze-target-until-close','--timeout','90','--output',str(ROOT/'supplement_replay')])
 args.training_manifest=None;args.max_joint_step=3.2;args.tracking_error_duration=100
 rclpy.init();node=Replay(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
 output=ROOT/'supplement_replay';output.mkdir(exist_ok=True);results=[]
 try:
  for number,folder in enumerate(folders,1):
   start=json.loads((folder/'episode_start.json').read_text());node.recorded_scene=start['scene_randomization'];node.frames=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()];node.index=0;node.playback_start=None
   result=node.run_episode('beverage_bottle',number,output,fixed_seed=node.recorded_scene['seed']);result.update(source_episode=str(folder),source_frames=len(node.frames),sent_frames=node.index)
   if result.get('error_detail')=='recorded_trajectory_complete':result['stop_reason']='recorded_trajectory_complete';result['error_detail']=''
   results.append(result);(output/'summary.json').write_text(json.dumps(results,indent=2));print('SUPPLEMENT_REPLAY_RESULT',number,result['success'],result['stop_reason'],flush=True)
 finally:
  node._set_active(False);executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
 eligible=[r['source_episode'] for r in results if r['success'] and r['physical_predicates'].get('physical_grasp_ok')]
 record.update(replay_completed=True,replay_eligible_episodes=eligible,training_eligible=bool(eligible),eligibility_reason='Only episodes with successful unassisted action replay are eligible; other episodes remain separate.')
 manifest.write_text(json.dumps(record,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
