"""ROS-side entry; use cli.py/SH to validate final-model gates first."""
import importlib.util
import json
from pathlib import Path
import sys

from contracts import HERE, RUN, check_bridge_verification, check_final, check_pins, training_coverage, validate_seeds, validate_rollout_contract


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Relative imports keep the tested ROS helpers; only this process gets our bridge.
load('z1_isaac_bridge.pi05_policy_bridge', HERE / 'policy_bridge.py')
runtime = load('z1_isaac_bridge.pi05_four_objects_runtime', HERE / 'runtime_rollout.py')


class FourObjectRollout(runtime.SmolVLASimRollout):
    def __init__(self, args):
        check_pins()
        checkpoint = check_final(require_reload=True)
        if args.checkpoint != checkpoint or args.policy_type != 'pi05':
            raise ValueError('Only this round final checkpoint is allowed')
        check_bridge_verification(checkpoint)
        validate_rollout_contract(args)
        coverage = training_coverage(RUN / 'split_manifest.json', HERE / 'grasp_physics.json')
        validate_seeds(args.eval_seeds, coverage)
        super().__init__(args)
        self.training_seeds = set(coverage['training_seeds'])
        self.covered_cells = {o: {tuple(c) for c in data['cells']} for o, data in coverage['objects'].items()}
        self.training_target_positions = {o: data['positions'] for o, data in coverage['objects'].items()}
        self.coverage = coverage

    def _status_callback(self, message):
        super()._status_callback(message)
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(status, dict):
            return
        with self._lock:
            for key in ('gripper_commanded_closed', 'inside_success_volume', 'placement_stable'):
                if isinstance(status.get(key), bool):
                    self._evidence[key] = self._evidence.get(key, False) or status[key]
            if isinstance(status.get('inside_success_volume'), bool):
                transported = status['inside_success_volume'] and bool(status.get('was_grasped') or self._evidence.get('physical_grasp_ok'))
                self._evidence['object_transported_to_box'] = self._evidence.get('object_transported_to_box', False) or transported

    def run_suite(self):
        from reporting import stage_summary
        root = super().run_suite()
        results = [json.loads(p.read_text()) for p in sorted((root / 'episodes').glob('*/result.json'))]
        (root / 'stage_report.json').write_text(json.dumps(stage_summary(results), indent=2) + '\n')
        provenance = dict(checkpoint=str(self.args.checkpoint), coverage=self.coverage,
                          source_pins=check_pins(), action_hz=10, execute_steps=self.args.execute_steps,
                          predicted_steps=30, inference_steps=10,
                          evaluation_kind='new seeds within training-covered cells; no held-out dataset')
        (root / 'evaluation_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return root


if __name__ == '__main__':
    runtime.SmolVLASimRollout = FourObjectRollout
    runtime.main()
