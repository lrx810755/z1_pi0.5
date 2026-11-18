"""Paired four-object PI0.5 evaluation with finger/table contact filtering."""
import json
import math
from pathlib import Path
import sys
import time

JOB = Path(__file__).resolve().parent
sys.path.insert(0, '/home/lp531/robot/pi05_four_objects_eval_20261117')
from entry import FourObjectRollout, runtime, load
from contracts import OBJECTS

# Keep the original checkpoint gates and all task rules. For fixed seeds,
# the local runtime uses the baseline scene's default settling parameters.
fixed_runtime = load('z1_isaac_bridge.filtered_table_runtime', JOB / 'runtime_rollout.py')
runtime.SmolVLASimRollout._randomize_scene = fixed_runtime.SmolVLASimRollout._randomize_scene


class FilteredTableRollout(FourObjectRollout):
    def __init__(self, args):
        if args.objects != OBJECTS or args.execute_steps != 15 or args.episodes_per_object != 30:
            raise ValueError('This run requires all four objects, 30 trials each, and execution window 15')
        physics = json.loads((JOB / 'physics_verified.json').read_text())
        if not physics.get('passed') or set(physics['support']) != set(args.objects):
            raise ValueError('Verified physical tabletop and box support required')
        self._baseline = json.loads((JOB / 'baseline.json').read_text())['objects']
        self._depth_log = None
        self._depth_samples = 0
        self._depth_max = [0.0, 0.0]
        super().__init__(args)
        if any(set(r['seed'] for r in rows) & self.training_seeds for rows in self._baseline.values()):
            raise ValueError('Unexpected training seed in paired evaluation')

    def _scene_grasp_callback(self, message):
        super()._scene_grasp_callback(message)
        status = json.loads(message.data)
        values = status.get('finger_table_penetration_estimate_m')
        if status.get('finger_table_filter_active') is not True or not isinstance(values, list) or len(values) != 2 or not all(math.isfinite(v) and v >= 0 for v in values):
            return
        with self._lock:
            if self._depth_log is None:
                return
            self._depth_log.write(json.dumps(dict(simulation_stamp_s=self._joint_sim_stamp,
                     left_depth_m=values[0], right_depth_m=values[1]))+'\n')
            self._depth_samples += 1
            self._depth_max = [max(a,b) for a,b in zip(self._depth_max,values)]

    def _episode_seed(self, target, prepared_seed, fixed_seed=None):
        for attempt in range(3):
            try:
                return super()._episode_seed(target, prepared_seed, fixed_seed)
            except (RuntimeError, TimeoutError) as error:
                if attempt == 2:
                    raise
                self.get_logger().warning(f'Retrying setup for {target}, seed {fixed_seed}: {error}')
                time.sleep(1)

    def run_episode(self, target, episode_number, root, prepared_seed=None, stop_event=None, fixed_seed=None):
        seed = self._baseline[target][episode_number-1]['seed']
        episode = root / 'episodes' / f'{target}_{episode_number:03d}_seed_{seed}'
        logs = root / 'table_penetration'
        logs.mkdir(exist_ok=True)
        depth_path = logs / f'{episode.name}.jsonl'
        with depth_path.open('x') as stream:
            with self._lock:
                self._depth_samples = 0
                self._depth_max = [0.0, 0.0]
                self._depth_log = stream
            try:
                result = super().run_episode(target, episode_number, root, stop_event=stop_event, fixed_seed=seed)
            finally:
                with self._lock:
                    self._depth_log = None
                    samples, maxima = self._depth_samples, self._depth_max[:]
        result.update(finger_table_collision_filtered=True, penetration_samples=samples,
                      max_finger_table_penetration_estimate_m=max(maxima) if samples else None,
                      per_finger_max_penetration_estimate_m=maxima if samples else None,
                      penetration_trace=str(depth_path))
        (episode / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        return result

    def run_suite(self):
        root = super().run_suite()
        path = root / 'evaluation_provenance.json'
        provenance = json.loads(path.read_text())
        provenance.update(evaluation_kind='matched seeds within training-covered cells',
                          collision_ablation='only two finger/tabletop collision pairs',
                          physics_verification=str(JOB/'physics_verified.json'),
                          baseline=str(JOB/'baseline.json'))
        path.write_text(json.dumps(provenance, indent=2)+'\n')
        return root


if __name__ == '__main__':
    runtime.SmolVLASimRollout = FilteredTableRollout
    runtime.main()
