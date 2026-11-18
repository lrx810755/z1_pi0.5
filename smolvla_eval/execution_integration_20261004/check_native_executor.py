"""Exercise the production executor in Isaac using the captured reference commands."""
import json,sys,threading
from pathlib import Path
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
sys.path.insert(0,'/home/lp531/smolvla_eval/pi05_four_step_validation_20261003')
from replay import Replay
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout,parse_args
BASE=Path(__file__).resolve().parent/'pause_preserved_final'
BASE.mkdir(exist_ok=False)
DATA=Path('/home/lp531/smolvla_eval/replay_debug_20261003/diagnostic_sparse_commands/episode_000000')
class ReferenceExecutor(Replay):
    _wait_sim_period=SmolVLASimRollout._wait_sim_period
    def _predict(self,batch):
        actions=self.frames[self.index:self.index+self.args.execute_steps]
        self.index+=len(actions)
        return np.array([r['action'] for r in actions] if actions else [self.frames[-1]['action']])

args=parse_args(['--checkpoint',str(DATA),'--objects','beverage_bottle','--execute-steps','5',
 '--execution-rate','60','--prelift-contact-stable-s','.5','--prelift-contact-timeout-s','2',
 '--freeze-target-until-close','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--timeout','40'])
args.training_manifest=None
rclpy.init();node=ReferenceExecutor(args);executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
try:
    node.frames=[json.loads(x) for x in (DATA/'frames.jsonl').read_text().splitlines()]
    node.recorded_scene=json.loads((DATA/'episode_start.json').read_text())['scene_randomization'];node.index=0
    result=node.run_episode('beverage_bottle',1,BASE,fixed_seed=node.recorded_scene['seed'])
    (BASE/'native_executor_result.json').write_text(json.dumps(result,indent=2))
    print('NATIVE_REFERENCE_RESULT',result['success'],result['stop_reason'],result['error_detail'],flush=True)
finally:
    node._set_active(False);executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
