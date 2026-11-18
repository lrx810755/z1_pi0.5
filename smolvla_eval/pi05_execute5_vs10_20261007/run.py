"""Eight paired simulator trials with full prediction/input recording."""
from pathlib import Path
from contextlib import nullcontext
import json,threading,time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout,parse_args
from z1_isaac_bridge.smolvla_rollout_core import execution_window
OUT=Path(__file__).resolve().parent
class RecordedRollout(SmolVLASimRollout):
    def _predict(self,batch):
        import torch
        started=time.perf_counter()
        with torch.inference_mode(),(torch.autocast(device_type='cuda',dtype=torch.float16) if self.use_amp else nullcontext()):
            chunk=self.policy.predict_action_chunk(batch)
        full=chunk.detach().float().cpu().numpy()
        inference_s=time.perf_counter()-started
        selected=execution_window(full,self.args.execute_steps)
        root=self._input_sync_path.parent
        index=self._input_sync_index-1
        archive=root/'policy_inputs';archive.mkdir(exist_ok=True)
        arrays={}
        for key,value in batch.items():
            if key.startswith('observation.images.'):
                arrays[key]=(value.detach().cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
            elif key=='observation.state':arrays[key]=value.detach().cpu().numpy()
        np.savez_compressed(archive/f'{index:04d}.npz',**arrays)
        with (root/'predictions.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(prediction=index,input_state=arrays['observation.state'].tolist(),task=batch['task'],full_actions=full[0].tolist(),execute_steps=self.args.execute_steps,inference_wall_s=inference_s,recording_wall_s=time.perf_counter()-started-inference_s,input_archive=f'policy_inputs/{index:04d}.npz'))+'\n')
        return selected
args=parse_args(['--policy-type','pi05','--checkpoint','/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints/080000/pretrained_model','--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--training-dataset-root','/home/lp531/smolvla_data/z1_tabletop_raw_left_wrist_three_camera_v1','--max-training-target-distance-m','.03','--reject-blank-cameras','--objects','beverage_bottle','--episodes-per-object','2','--eval-seeds','489361075','774196979','--rate','10','--execution-rate','60','--prelift-contact-stable-s','0','--binary-gripper','--gripper-action-encoding','binary','--no-freeze-target-until-close'])
plan=[(5,1),(10,1),(10,2),(5,2)]
(OUT/'plan.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),seeds=args.eval_seeds,inference_seed=123,trials=8,order=plan,timeout_sim_s=args.timeout,logging='full15 actions and lossless actual policy input arrays; same controller, includes logging overhead'),indent=2)+'\n')
rclpy.init()
node=RecordedRollout(args)
executor=MultiThreadedExecutor(num_threads=4);executor.add_node(node)
thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
completed=[]
try:
    for steps,repeat in plan:
        args.execute_steps=steps;args.output=OUT/f'steps{steps}_repeat{repeat}'
        print('BEGIN',steps,repeat,flush=True)
        root=node.run_suite();completed.append(str(root))
        (OUT/'progress.json').write_text(json.dumps(dict(completed_suites=completed),indent=2)+'\n')
        results=[json.loads(p.read_text()) for p in root.rglob('result.json')]
        if any(r['failure_category']=='episode_setup_failed' for r in results):
            raise RuntimeError('Scene setup failed; inspect results before continuing paired experiment')
        print('END',steps,repeat,[(r['seed'],r['failure_category']) for r in results],flush=True)
finally:
    node._set_active(False)
    executor.shutdown();thread.join(timeout=2)
    node.policy.close();node.destroy_node();rclpy.shutdown()
print('COMPLETE',flush=True)
