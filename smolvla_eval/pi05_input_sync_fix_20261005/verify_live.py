"""Verify production input building against live ROS without sending motion."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
import torch
from rclpy.executors import MultiThreadedExecutor

from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args

ROOT=Path(__file__).resolve().parent

class PassiveRollout(SmolVLASimRollout):
    def _load_policy(self, checkpoint):
        self.camera_keys=['wrist','global']
        self.state_dim=7
        self.device=torch.device('cpu')
        self.use_amp=False
        return SimpleNamespace(reset=lambda:None)

def main():
    args=parse_args(['--policy-type','pi05','--checkpoint','/home/lp531/smolvla_training/pi05_bottle_lora_20261001/checkpoints/108000/pretrained_model',
                     '--objects','beverage_bottle','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--reject-blank-cameras'])
    args.training_manifest=None
    rclpy.init();node=PassiveRollout(args)
    node._input_sync_path=ROOT/'live_input_sync.jsonl'
    if node._input_sync_path.exists():raise RuntimeError('Refuse to overwrite verification')
    executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start();samples=[];errors=[]
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            try:node._snapshot();break
            except RuntimeError:time.sleep(.05)
        started=time.monotonic()
        while time.monotonic()-started<20:
            tick=time.monotonic()
            try:
                images,joints,status,_=node._snapshot()
                batch=node._policy_batch(images,joints,status,'Physically grasp the bottle, lift it clear, and place it in the box.')
                assert batch['observation.state'].shape==(1,7)
                assert batch['observation.images.wrist'].shape==(1,3,360,640)
                assert batch['observation.images.global'].shape==(1,3,480,640)
                assert np.isfinite(batch['observation.state'].numpy()).all()
                samples.append(dict(node._last_input_sync))
            except RuntimeError as exc:errors.append(str(exc))
            time.sleep(max(0,.1-(time.monotonic()-tick)))
    finally:
        executor.shutdown();thread.join(timeout=2);node.destroy_node();rclpy.shutdown()
    assert len(samples)>=30, {'samples':len(samples),'errors':errors}
    result={'scope':'20-second passive live production _policy_batch; no model inference or robot commands',
            'samples':len(samples),'errors':errors,'settings':{'max_skew_s':args.input_sync_tolerance,'max_age_s':args.input_sync_max_age,'wait_timeout_s':args.input_sync_timeout}}
    for key in ['sensor_skew_s','latest_received_sensor_skew_s','oldest_sensor_age_s','wait_wall_s']:
        values=[s[key] for s in samples]
        result[key]={'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'maximum':float(max(values))}
    assert result['sensor_skew_s']['maximum']<=args.input_sync_tolerance+1e-9
    assert result['oldest_sensor_age_s']['maximum']<=args.input_sync_max_age+1e-9
    (ROOT/'live_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
