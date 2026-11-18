import json,sys,threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
sys.path.insert(0,'/home/lp531/smolvla_eval/pi05_assisted_grasp_20261007')
from run import Assisted
from z1_isaac_bridge.smolvla_sim_rollout import parse_args
ROOT=Path(__file__).resolve().parent
class Replay(Assisted):
 def _load_policy(self,checkpoint):
  self.camera_keys=['wrist','global','global_right'];self.state_dim=7
  return SimpleNamespace(reset=lambda:None,close=lambda:None)
 def _policy_batch(self,*args):return None
 def _predict(self,batch):
  i=min(self.index,len(self.actions)-1);a=self.actions[i].copy()
  if self.args.binary_gripper:a[-1]=float(self.label['close_frame']<=i<self.label['open_frame'])
  self.index+=1
  if self.index>len(self.actions)+50:raise RuntimeError('reference_complete_without_success')
  return a[None]
args=parse_args(['--checkpoint',str(ROOT),'--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--reject-blank-cameras','--execute-steps','1','--rate','10','--execution-rate','60','--prelift-contact-stable-s','0','--no-freeze-target-until-close','--binary-gripper','--gripper-action-encoding','binary','--timeout','100','--output',str(ROOT)])
args.training_manifest=None
rclpy.init();node=Replay(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node);thread=threading.Thread(target=executor.spin,daemon=True);thread.start();results=[]
try:
 for label in json.loads((ROOT/'selected.json').read_text()):
  source=Path(label['raw_root']);node.label=label;node.recorded_scene=json.loads((source/'episode_start.json').read_text())['scene_randomization'];node.actions=np.array([json.loads(x)['action'] for x in (source/'frames.jsonl').read_text().splitlines()])
  for mode in ['binary']:
   if mode=='original_if_failed' and results[-1]['success']:break
   args.binary_gripper=mode=='binary';node.index=0
   path=ROOT/'replays'/f'{source.name}_{mode}';path.mkdir(parents=True,exist_ok=False)
   print('BEGIN',source.name,mode,flush=True)
   result=node.run_episode('beverage_bottle',len(results)+1,path,fixed_seed=label['seed']);result.update(source_episode=source.name,mode=mode,frames_sent=node.index)
   expected=np.array(node.recorded_scene['object_positions_m']['beverage_bottle']);actual=np.array(result['target_position_m']);result['layout_error_m']=float(np.linalg.norm(expected-actual))
   results.append(result);(ROOT/'replay_results.json').write_text(json.dumps(results,indent=2));print('RESULT',source.name,mode,result['success'],result['stop_reason'],result['error_detail'],flush=True)
finally:
 node._set_active(False);executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
(ROOT/'replay_complete.json').write_text(json.dumps(dict(trials=len(results),successes=sum(x['success'] for x in results))))
