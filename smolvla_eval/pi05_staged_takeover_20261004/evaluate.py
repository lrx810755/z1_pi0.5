"""Fixed-layout stage handoffs using production inference and execution safeguards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
sys.path.insert(0, '/home/lp531/smolvla_eval/pi05_four_step_validation_20261003')
from replay import Replay
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args
from z1_isaac_bridge.smolvla_rollout_core import GraspPhaseTracker, JOINT_NAMES, safe_joint_target, execution_window

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/lp531/smolvla_eval/replay_debug_20261003/diagnostic_sparse_commands/episode_000000')
CHECKPOINT = Path('/home/lp531/smolvla_training/pi05_bottle_lora_20261001/checkpoints/108000/pretrained_model')
STAGES = {'home': 0, 'midway': 50, 'preclose': 105}
SEED = 342470491


def prefix_target(actions, elapsed):
    index = min(elapsed * 10, len(actions) - 1)
    lo = int(index)
    hi = min(lo + 1, len(actions) - 1)
    target = actions[lo] + (actions[hi] - actions[lo]) * (index - lo)
    if target[-1] > -.85:
        raise ValueError('guided prefix must keep gripper fully open')
    return target


class StagedEvaluation(Replay):
    _load_policy = SmolVLASimRollout._load_policy
    _policy_batch = SmolVLASimRollout._policy_batch
    _wait_sim_period = SmolVLASimRollout._wait_sim_period

    def _predict(self, batch):
        if self.stage == 'reference':
            actions = self.reference[self.index:self.index + self.args.execute_steps]
            self.index += len(actions)
            return actions if len(actions) else self.reference[-1:]
        import torch
        with self._lock:
            sim_before = self._joint_sim_stamp
        started = time.monotonic()
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch).detach().float().cpu().numpy()
        with self._lock:
            sim_after = self._joint_sim_stamp
        record = {'prediction': self.prediction_index, 'simulation_start': sim_before,
                  'simulation_end': sim_after, 'wall_duration_s': time.monotonic() - started,
                  'input_state': batch['observation.state'].detach().cpu().tolist(),
                  'actions': chunk.tolist()}
        with (self.trial_dir / 'full_predictions.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.prediction_index += 1
        return execution_window(chunk, self.args.execute_steps)

    def _prepare_episode(self, target, fixed_seed=None):
        seed = super()._prepare_episode(target, fixed_seed)
        if self.stage == 'reference':
            return seed
        self._set_target_frozen(target, True)
        self._contact_gate = None
        tracker = GraspPhaseTracker()
        with self._lock:
            start = self._joint_sim_stamp
        end_index = STAGES[self.stage]
        previous = None
        error_since = None
        prefix_rows = []
        # Half a simulation second at the endpoint gives every group fresh live images.
        for tick in range(round((end_index / 10 + .5) * 60) + 1):
            if tick:
                self._wait_sim_period(start + (tick - 1) / 60)
            images, current, status, _ = self._snapshot()
            with self._lock:
                sim = self._joint_sim_stamp
            if status.get('left_finger_touching_table') or status.get('right_finger_touching_table'):
                raise RuntimeError('prefix_finger_table_contact')
            if previous is not None:
                error = float(np.max(np.abs(previous[:6] - current[:6])))
                error_since = (sim if error_since is None else error_since) if error > .08 else None
                if error > .20 or (error_since is not None and sim - error_since > .20):
                    raise RuntimeError('prefix_joint_tracking_fault')
            elapsed = min(tick / 60, end_index / 10)
            requested = prefix_target(self.reference[:end_index + 1], elapsed)
            command = safe_joint_target(requested, current, self.args.max_joint_step)
            self._publish_grasp_phase(target, tracker.update(command[-1], current[-1], status))
            self._publish_command(command)
            previous = command
            prefix_rows.append({'elapsed_s': sim - start, 'command': command.tolist(),
                                'joint_state': current.tolist(), 'status': status})
        (self.trial_dir / 'prefix_trace.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prefix_rows))
        images, current, status, _ = self._snapshot()
        arm_error = float(np.max(np.abs(current[:6] - self.reference[end_index, :6])))
        if current[-1] > -.85 or arm_error > .02:
            raise RuntimeError(f'invalid_handoff_pose: jaw={current[-1]}, arm_error={arm_error}')
        if not status.get('frozen') or status.get('target_collisions_enabled') is not True:
            raise RuntimeError('invalid_handoff_physics')
        if self.stage == 'preclose' and not status.get('inside_grasp_envelope'):
            raise RuntimeError('preclose_pose_outside_grasp_envelope')
        self.policy.reset()
        with self._lock:
            self._evidence = {}
        for name, image in images.items():
            cv2.imwrite(str(self.trial_dir / f'handoff_{name}.png'), image)
        self.handoff = {'source_index': end_index, 'reference_joints': self.reference[end_index].tolist(),
                        'actual_joints': current.tolist(), 'arm_error_rad': arm_error,
                        'status': status, 'scene': dict(self._randomization_status)}
        (self.trial_dir / 'handoff.json').write_text(json.dumps(self.handoff, indent=2))
        print('HANDOFF', self.stage, self.repeat, 'jaw', current[-1],
              'relative', status.get('relative_position_gripper_m'), flush=True)
        return seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    opts = parser.parse_args()
    reference = np.array([json.loads(x)['action'] for x in (DATA / 'frames.jsonl').read_text().splitlines()])
    assert reference.shape[1] == 7
    for end in STAGES.values():
        assert np.all(reference[:end + 1, -1] <= -.85)
        assert np.allclose(prefix_target(reference[:end + 1], end / 10), reference[end])
    assert np.allclose(prefix_target(reference, .05), (reference[0] + reference[1]) / 2)
    if opts.check:
        print('CHECK PASS: all guided prefixes stay open; interpolation endpoints and midpoint checked')
        return
    os.environ['Z1_PI05_INFERENCE_SEED'] = '20261004'
    args = parse_args(['--policy-type', 'pi05', '--checkpoint', str(CHECKPOINT), '--objects', 'beverage_bottle',
                      '--execute-steps', '5', '--rate', '10', '--execution-rate', '60',
                      '--prelift-contact-stable-s', '.5', '--prelift-contact-timeout-s', '2',
                      '--freeze-target-until-close', '--wrist-rgb-topic', '/left_wrist_camera/color/image_raw',
                      '--reject-blank-cameras', '--timeout', '60', '--output', str(ROOT)])
    args.training_manifest = None
    if (ROOT / 'started.json').exists():
        raise RuntimeError('Existing run: refusing to overwrite experiment evidence')
    (ROOT / 'started.json').write_text(json.dumps({'scene_seed': SEED, 'grid': [3, 0], 'stages': STAGES,
        'repetitions': 5, 'inference_seed': 20261004, 'checkpoint': str(CHECKPOINT),
        'source_sha256': hashlib.sha256((DATA / 'frames.jsonl').read_bytes()).hexdigest(),
        'execution_args': vars(args)}, default=str, indent=2))
    rclpy.init()
    node = None
    executor = None
    results = []
    try:
        node = StagedEvaluation(args)
        node.reference = reference
        node.recorded_scene = json.loads((DATA / 'episode_start.json').read_text())['scene_randomization']
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        schedule = [('reference', k) for k in range(1, 4)]
        schedule += [(stage, repeat) for repeat in range(1, 6) for stage in STAGES]
        reference_passed = False
        for stage, repeat in schedule:
            if stage == 'reference' and reference_passed:
                continue
            if stage != 'reference' and not reference_passed:
                raise RuntimeError('reference_control_failed_three_times')
            node.stage, node.repeat, node.index, node.prediction_index = stage, repeat, 0, 0
            node.trial_dir = ROOT / stage / f'trial_{repeat:02d}'
            node.trial_dir.mkdir(parents=True, exist_ok=False)
            print('START', stage, repeat, flush=True)
            result = node.run_episode('beverage_bottle', repeat, node.trial_dir, fixed_seed=SEED)
            result.update(stage=stage, repetition=repeat, handoff=getattr(node, 'handoff', None))
            results.append(result)
            (ROOT / 'results.json').write_text(json.dumps(results, indent=2))
            print('RESULT', stage, repeat, result['success'], result['stop_reason'], result['error_detail'], flush=True)
            if stage == 'reference':
                reference_passed = result['success']
            if result['stop_reason'] in {'runtime_error', 'stale_stream', 'stale_grasp_status', 'blank_camera', 'target_unfreeze_failed', 'invalid_policy_action'}:
                raise RuntimeError('infrastructure_failure: ' + result['error_detail'])
        (ROOT / 'completed.json').write_text(json.dumps({'model_trials': sum(r['stage'] != 'reference' for r in results),
                                                         'reference_passed': reference_passed}, indent=2))
    finally:
        if node:
            try:
                with node._lock:
                    hold = [node._joints.get(name) for name in JOINT_NAMES]
                if all(x is not None for x in hold):
                    node._publish_command(hold)
                node._set_target_frozen('beverage_bottle', False)
            finally:
                node._set_active(False)
        if executor:
            executor.shutdown()
            thread.join(timeout=2)
        if node:
            node.policy.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
