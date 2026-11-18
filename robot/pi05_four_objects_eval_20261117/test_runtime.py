"""Real CPU integration: ROS parser, process transport, and source coverage."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import os
import subprocess
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class RuntimeTests(unittest.TestCase):
    def test_execution_windows_10_25_20_30_preserve_30_step_prediction(self):
        import numpy as np
        from contracts import FINAL, parse_args, rollout_arguments, validate_rollout_contract
        from entry import runtime
        for steps in [10, 25, 20, 30]:
            args = parse_args(['--run', '--execute-steps', str(steps), '--episodes-per-object', '25'])
            parsed = runtime.parse_args(rollout_arguments(args, FINAL))
            validate_rollout_contract(parsed)
            self.assertEqual(parsed.execute_steps, steps)
            prediction = np.arange(210).reshape(1, 30, 7)
            selected = runtime.execution_window(prediction, parsed.execute_steps)
            self.assertEqual(selected.shape, (steps, 7))
            self.assertEqual(selected[-1, -1], steps * 7 - 1)
            self.assertEqual(prediction.shape, (1, 30, 7))
        with self.assertRaises(SystemExit):
            parse_args(['--execute-steps', '31'])

    def test_ros_conda_environment_can_import_model_dependencies(self):
        from cli import command_environment
        from contracts import ROS_PYTHON
        with patch.dict(os.environ, {'PYTHONNOUSERSITE': '1'}):
            result = subprocess.run([ROS_PYTHON, '-c', 'import torch, typing_extensions, cv2, rclpy'],
                                    env=command_environment('run'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_deployment_request_excludes_training_action_tensor(self):
        import torch
        from policy_bridge import encode_batch
        from verify_bridge import deployment_input
        batch = {'observation.state': torch.zeros(1, 7), 'task': ['test'], 'action': torch.zeros(1, 30, 7)}
        payload = json.loads(json.dumps(encode_batch(deployment_input(batch))))
        self.assertEqual(set(payload), {'observation.state', 'task'})
        self.assertEqual(payload['observation.state'], [[0.0] * 7])

    def test_all_objects_use_real_rollout_parser(self):
        from contracts import FINAL, parse_args, rollout_arguments
        from entry import runtime
        for targets in [['apple'], ['peach'], ['cup'], ['bottle'], ['all']]:
            args = parse_args(['--objects', *targets])
            parsed = runtime.parse_args(rollout_arguments(args, FINAL))
            self.assertEqual(parsed.objects, args.objects)
            self.assertEqual(parsed.execute_steps, 15)
            self.assertEqual(parsed.rate, 10)
            self.assertIsNone(parsed.training_manifest)

    def test_real_frozen_manifest_coverage(self):
        from contracts import RUN, TASKS, training_coverage, validate_seeds
        coverage = training_coverage(RUN / 'split_manifest.json', HERE / 'grasp_physics.json')
        self.assertEqual(coverage['records'], 1762)
        self.assertEqual(set(coverage['objects']), set(TASKS))
        for data in coverage['objects'].values():
            self.assertGreater(len(data['positions']), 300)
            self.assertGreater(len(data['cells']), 10)
        with self.assertRaises(ValueError):
            validate_seeds([coverage['training_seeds'][0]], coverage)

    def test_raw_three_camera_rgb_transport(self):
        import numpy as np
        import torch
        from policy_bridge import encode_batch, decode_batch, validate_batch
        from contracts import TASKS
        batch = {'observation.state': torch.tensor([[.1, .2, -.3, .4, -.5, .6, -.7]]), 'task': [TASKS['peach']]}
        for c, height in [('wrist', 360), ('global', 480), ('global_right', 480)]:
            pixels = np.random.default_rng(height).integers(0, 256, (1, 3, height, 640), dtype=np.uint8)
            batch['observation.images.' + c] = torch.from_numpy(pixels).float() / 255
        restored = decode_batch(encode_batch(batch))
        validate_batch(restored)
        for key in batch:
            if key == 'task':
                self.assertEqual(restored[key], batch[key])
            else:
                self.assertTrue(torch.equal(restored[key], batch[key]))
        restored['observation.images.wrist'] = restored['observation.images.wrist'][:, :, :224, :224]
        with self.assertRaisesRegex(ValueError, 'camera'):
            validate_batch(restored)

    def test_stage_summary_distinguishes_unknown_from_failure(self):
        from reporting import stage_summary
        results = [dict(target_object='apple', success=False, physical_predicates={'bilateral_contact_ever': True},
                        final_status={'object_lifted_clear_of_table_ever': True}),
                   dict(target_object='apple', success=True, physical_predicates={}, final_status={})]
        summary = stage_summary(results)['apple']
        self.assertEqual(summary['task_success']['true'], 1)
        self.assertEqual(summary['bilateral_contact']['true'], 1)
        self.assertEqual(summary['bilateral_contact']['unknown'], 1)
        self.assertEqual(summary['lifted']['true'], 1)


if __name__ == '__main__':
    unittest.main()
