import json,os,threading,time
from pathlib import Path
import rclpy
from rclpy.executors import MultiThreadedExecutor
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout,parse_args
ROOT=Path(__file__).resolve().parent
CHECKPOINTS={
'baseline':'/home/lp531/smolvla_training/pi05_bottle_lora_20261001/checkpoints/108000/pretrained_model',
'uniform':'/home/lp531/smolvla_training/pi05_bottle_sampling_ab_20261003/uniform/checkpoints/012000/pretrained_model',
'weighted':'/home/lp531/smolvla_training/pi05_bottle_sampling_ab_20261003/weighted/checkpoints/012000/pretrained_model'}
SEEDS=[882884764,1370963773,763856373]
def main():
 os.environ['Z1_PI05_INFERENCE_SEED']='20261003'
 for arm,checkpoint in CHECKPOINTS.items():
  output=ROOT/'corrected_comparison'/arm;output.mkdir(parents=True,exist_ok=True)
  if (output/'completed.json').exists():continue
  args=parse_args(['--policy-type','pi05','--checkpoint',checkpoint,'--objects','beverage_bottle','--episodes-per-object','9','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--training-dataset-root','/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1','--max-training-target-distance-m','0.03','--reject-blank-cameras','--execute-steps','5','--freeze-target-until-close','--timeout','60','--output',str(output)])
  (output/'specification.json').write_text(json.dumps({'arm':arm,'checkpoint':checkpoint,'scene_seeds':SEEDS,'repetitions':3,'inference_seed':20261003,'physics':'collision_preserving_fixed_until_close','note':'same sampling seed across repeated trials; repetitions measure physics/reset variation'},indent=2))
  rclpy.init();node=None;executor=None;thread=None;results=[]
  try:
   node=SmolVLASimRollout(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
   for repeat in range(3):
    for layout,seed in enumerate(SEEDS):
     number=repeat*3+layout+1
     result=node.run_episode('beverage_bottle',number,output,fixed_seed=seed)
     result['repetition']=repeat+1;results.append(result)
     (output/'summary.json').write_text(json.dumps(results,indent=2))
     print('CORRECTED_RESULT',arm,repeat+1,seed,result['success'],result['stop_reason'],flush=True)
     if result['stop_reason'] in {'runtime_error','stale_stream','blank_camera','target_unfreeze_failed','invalid_policy_action'}: raise RuntimeError(result.get('error_detail',result['stop_reason']))
   (output/'completed.json').write_text(json.dumps({'arm':arm,'episodes':len(results),'successes':sum(r['success'] for r in results)},indent=2))
  finally:
   if node:node._set_active(False)
   if executor:executor.shutdown()
   if thread:thread.join(timeout=2)
   if node:node.policy.close();node.destroy_node()
   if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
