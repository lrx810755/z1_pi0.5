"""Live three-camera model closure with recorded MoveIt arm assistance."""
import json, sys, threading, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args
ROOT = Path(__file__).resolve().parent
LABELS = json.loads(Path('/home/lp531/pi05_lora_workspace/binary_gripper_20261007/episode_labels.json').read_text())

class Assisted(SmolVLASimRollout):
    def _randomize_scene(self, target, fixed_seed=None):
        original = self._randomization_command_pub
        grid = self.recorded_scene.get('target_grid')
        class Publisher:
            def publish(self, message):
                payload = json.loads(message.data)
                payload['target_grid'] = grid
                message.data = json.dumps(payload)
                original.publish(message)
        self._randomization_command_pub = Publisher()
        try:
            return super()._randomize_scene(target, fixed_seed)
        finally:
            self._randomization_command_pub = original

    def _policy_batch(self, images, joints, status, task):
        if self.mode == 'assisted' and self.index == self.label['close_frame']:
            return super()._policy_batch(images, joints, status, task)
        return None

    def _predict(self, batch):
        close, opened = self.label['close_frame'], self.label['open_frame']
        if self.mode == 'assisted' and self.index == close:
            import torch
            _, current, status, _ = self._snapshot()
            pose_error = float(np.max(np.abs(current[:6] - self.actions[close-1,:6])))
            relative = status.get('relative_position_gripper_m')
            if pose_error > .025 or not relative or not (.10 < relative[0] < .17 and abs(relative[1]) < .025 and abs(relative[2]-.019) < .025):
                raise RuntimeError('invalid_handoff_geometry: '+json.dumps(dict(error=pose_error, relative=relative)))
            if status.get('frozen'):
                raise RuntimeError('unexpected_frozen_bottle')
            started = time.monotonic()
            with torch.inference_mode():
                full = self.policy.predict_action_chunk(batch).detach().float().cpu().numpy()[0]
            indices = np.flatnonzero(full[:,-1] >= .5)
            record = dict(input_state=batch['observation.state'].tolist(), actions=full.tolist(),
                          first_close_index=int(indices[0]) if len(indices) else None,
                          inference_s=time.monotonic()-started, status=status, arm_pose_error_rad=pose_error,
                          arm_execution='held at accepted demonstration grasp pose')
            (self.trial/'model_closure.json').write_text(json.dumps(record, indent=2))
            np.savez_compressed(self.trial/'model_input.npz', **{k:v.detach().cpu().numpy() for k,v in batch.items() if hasattr(v,'detach')})
            print('MODEL_CLOSURE', record['first_close_index'], 'relative',relative,flush=True)
            if not len(indices):
                self.attempts += 1
                if self.attempts >= 5:
                    raise RuntimeError('model_did_not_predict_close_in_15_frames')
                action = self.actions[close-1].copy(); action[-1]=0
                return action[None]
        if self.index >= len(self.actions):
            self.tail += 1
            if self.tail > 50:
                raise RuntimeError('reference_complete_without_task_success')
            action = self.actions[-1].copy()
        else:
            action = self.actions[self.index].copy()
        if self.mode == 'assisted':
            action[-1] = float(close <= self.index < opened)
            if self.index > close and self.index < opened and not self.lift_started:
                _, current, status, evidence = self._snapshot()
                # Isolated proof lift starts only after actual bilateral contact.
                if status.get('bilateral_contact') or (status.get('left_finger_contact_object') and status.get('right_finger_contact_object')):
                    self.lift_started = True
                    print('BILATERAL_CONTACT',self.index,flush=True)
                elif np.max(np.abs(action[:6]-self.actions[close,:6])) > .003:
                    self.contact_attempts += 1
                    if self.contact_attempts > 30:
                        raise RuntimeError('bilateral_contact_not_established')
                    action[:6] = self.actions[close,:6]
                    return action[None]
        self.index += 1
        return action[None]

def main():
    args = parse_args(['--policy-type','pi05','--checkpoint','/home/lp531/smolvla_training/pi05_bottle_binary_chunk15_attn_ffn_20261007/checkpoints/080000/pretrained_model',
                      '--wrist-rgb-topic','/left_wrist_camera/color/image_raw','--reject-blank-cameras',
                      '--objects','beverage_bottle','--execute-steps','1','--rate','10','--execution-rate','60',
                      '--prelift-contact-stable-s','0','--no-freeze-target-until-close','--timeout','100','--output',str(ROOT)])
    args.training_manifest = None
    rclpy.init(); node = None; executor = None; results=[]
    try:
        node = Assisted(args)
        executor=MultiThreadedExecutor(num_threads=4); executor.add_node(node)
        thread=threading.Thread(target=executor.spin,daemon=True); thread.start()
        for label in LABELS[:3]:
            source=Path(label['raw_root'])
            node.label=label
            node.recorded_scene=json.loads((source/'episode_start.json').read_text())['scene_randomization']
            node.actions=np.array([json.loads(x)['action'] for x in (source/'frames.jsonl').read_text().splitlines()])
            for mode in ['reference','assisted']:
                node.mode=mode;node.index=0;node.attempts=0;node.tail=0;node.contact_attempts=0;node.lift_started=False
                args.binary_gripper=mode=='assisted';args.gripper_action_encoding='binary' if args.binary_gripper else 'angle'
                node.trial=ROOT/f'{source.name}_{mode}';node.trial.mkdir(exist_ok=False)
                print('BEGIN',source.name,mode,flush=True)
                result=node.run_episode('beverage_bottle',len(results)+1,node.trial,fixed_seed=node.recorded_scene['seed'])
                result.update(mode=mode,source_episode=str(source),model_controls='closure decision only' if mode=='assisted' else 'none',arm_controls='recorded MoveIt demonstration',sent_reference_frames=node.index)
                results.append(result);(ROOT/'results.json').write_text(json.dumps(results,indent=2))
                print('RESULT',mode,result['success'],result['stop_reason'],result['error_detail'],result['physical_predicates'],flush=True)
                if mode=='reference' and not result['success']:
                    break
                if mode=='assisted' and result['success']:
                    (ROOT/'completed.json').write_text(json.dumps(dict(success=True,assisted=True)))
                    return
        (ROOT/'completed.json').write_text(json.dumps(dict(success=False,assisted=True)))
    finally:
        if node: node._set_active(False)
        if executor: executor.shutdown();thread.join(timeout=2)
        if node: node.policy.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
