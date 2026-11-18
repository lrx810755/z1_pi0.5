"""Read live synchronized inputs and predict with PI05; never publish commands."""
import json
import threading
import time
from pathlib import Path
import torch
import rclpy
from rclpy.executors import MultiThreadedExecutor
from verify_live import PassiveRollout
from z1_isaac_bridge.smolvla_sim_rollout import parse_args
from z1_isaac_bridge.pi05_policy_bridge import PI05ProcessPolicy

ROOT=Path(__file__).resolve().parent
CHECKPOINT=Path('/home/lp531/smolvla_training/pi05_bottle_lora_20261001/checkpoints/108000/pretrained_model')
args=parse_args(['--policy-type','pi05','--checkpoint',str(CHECKPOINT),'--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--reject-blank-cameras'])
args.training_manifest=None
rclpy.init()
node=PassiveRollout(args)
node._input_sync_path=ROOT/'model_input_sync.jsonl'
assert not node._input_sync_path.exists()
executor=MultiThreadedExecutor(num_threads=4)
executor.add_node(node)
thread=threading.Thread(target=executor.spin,daemon=True)
thread.start()
policy=None
results=[]
try:
    policy=PI05ProcessPolicy(CHECKPOINT)
    policy.reset()
    for index in range(3):
        images,joints,status,_=node._snapshot()
        batch=node._policy_batch(images,joints,status,'Physically grasp the bottle, lift it clear, and place it in the box.')
        started=time.monotonic()
        actions=policy.predict_action_chunk(batch)
        assert tuple(actions.shape)==(1,50,7), actions.shape
        assert torch.isfinite(actions).all()
        result={'prediction':index,'latency_wall_s':time.monotonic()-started,'action_shape':list(actions.shape),'finite':True,'input_sync':dict(node._last_input_sync)}
        results.append(result)
        print(json.dumps(result),flush=True)
finally:
    if policy is not None:policy.close()
    executor.shutdown()
    thread.join(timeout=2)
    node.destroy_node()
    rclpy.shutdown()
assert len(results)==3
(ROOT/'model_summary.json').write_text(json.dumps({'scope':'Three real PI05 predictions from synchronized live ROS input; no motion commands','checkpoint':str(CHECKPOINT),'results':results},indent=2))
