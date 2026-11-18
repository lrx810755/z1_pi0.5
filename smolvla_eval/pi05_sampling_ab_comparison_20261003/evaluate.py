"""Replay ten bottle layouts with the existing live Isaac evaluation runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {
    'baseline': '/home/lp531/smolvla_training/pi05_bottle_lora_20261001/checkpoints/108000/pretrained_model',
    'uniform': '/home/lp531/smolvla_training/pi05_bottle_sampling_ab_20261003/uniform/checkpoints/012000/pretrained_model',
    'weighted': '/home/lp531/smolvla_training/pi05_bottle_sampling_ab_20261003/weighted/checkpoints/012000/pretrained_model',
}
PRIOR_FAILURE_SEEDS = [763856373, 20855437, 1001730107, 393805101, 882884764]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--arm', choices=CHECKPOINTS, required=True)
    arm = cli.parse_args().arm
    marker = ROOT / f'{arm}_completed.json'
    if marker.exists():
        print(f'Already completed: {arm}', flush=True)
        return
    checkpoint = Path(CHECKPOINTS[arm])
    common = [
        '--policy-type', 'pi05', '--checkpoint', str(checkpoint),
        '--objects', 'beverage_bottle', '--episodes-per-object', '10',
        '--wrist-rgb-topic', '/left_wrist_camera/color/image_raw',
        '--training-dataset-root', '/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1',
        '--max-training-target-distance-m', '0.03', '--reject-blank-cameras',
        '--execute-steps', '5', '--freeze-target-until-close', '--timeout', '60',
        '--output', str(ROOT / arm),
    ]
    if arm != 'baseline':
        seeds = json.loads((ROOT / 'paired_seeds.json').read_text())
        common += ['--eval-seeds', *map(str, seeds)]
    args = parse_args(common)
    if arm == 'baseline':
        # Reproduce earlier failures, then let the native coverage filter choose five new layouts.
        args.eval_seeds = PRIOR_FAILURE_SEEDS + [None] * 5
    os.environ['Z1_PI05_INFERENCE_SEED'] = '20261003'
    specification = {
        'arm': arm, 'checkpoint': str(checkpoint), 'arguments': common,
        'requested_scene_seeds': args.eval_seeds, 'inference_seed_per_episode': 20261003,
        'adapter_sha256': hashlib.sha256((checkpoint / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        'comparison': 'same final update count for A/B; same scene layouts and control settings',
    }
    (ROOT / f'{arm}_specification.json').write_text(json.dumps(specification, indent=2))
    rclpy.init()
    node = None
    executor = None
    thread = None
    try:
        node = SmolVLASimRollout(args)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        output = node.run_suite()
        results = [json.loads(p.read_text()) for p in sorted((output / 'episodes').glob('*/result.json'))]
        if len(results) != 10 or any(r['seed'] is None for r in results):
            raise RuntimeError(f'Incomplete paired evaluation; inspect {output}')
        seeds = [r['seed'] for r in sorted(results, key=lambda r: r['episode_number'])]
        if arm == 'baseline':
            (ROOT / 'paired_seeds.json').write_text(json.dumps(seeds, indent=2))
        elif seeds != json.loads((ROOT / 'paired_seeds.json').read_text()):
            raise RuntimeError('Scene seeds do not match baseline')
        if any(r['stop_reason'] in {'stale_stream', 'blank_camera', 'runtime_error', 'target_unfreeze_failed', 'invalid_policy_action'} for r in results):
            raise RuntimeError(f'Infrastructure/policy error in {output}; inspect before interpreting')
        marker.write_text(json.dumps({'arm': arm, 'output': str(output), 'episodes': 10, 'scene_seeds': seeds}, indent=2))
        print('EVALUATION_COMPLETE', arm, output, flush=True)
    finally:
        if node is not None and rclpy.ok():
            node._set_active(False)
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2)
        if node is not None:
            node.policy.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
