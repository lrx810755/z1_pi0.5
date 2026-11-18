import json
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO

HERE = Path(__file__).resolve().parent
EXAMPLE = Path('/home/lp531/smolvla_training/pi05_current_round_20261111/checkpoints/760000/pretrained_model')


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.output = self.root / 'output'
        self.checkpoint = self.output / 'checkpoints/800000/pretrained_model'
        self.checkpoint.mkdir(parents=True)
        for name in ['config.json', 'adapter_config.json', 'policy_preprocessor.json', 'policy_postprocessor.json']:
            shutil.copyfile(EXAMPLE / name, self.checkpoint / name)
        for name in ['adapter_model.safetensors', 'policy_preprocessor_step_4_normalizer_processor.safetensors',
                     'policy_postprocessor_step_0_unnormalizer_processor.safetensors']:
            (self.checkpoint / name).write_bytes(b'fixture')
        (self.checkpoint / 'tokenizer').mkdir()
        (self.checkpoint / 'tokenizer/tokenizer.json').write_text('{}')
        reference = self.run / 'processor_verification'
        reference.mkdir()
        for path in self.checkpoint.glob('policy_*'):
            shutil.copyfile(path, reference / path.name)
        self.write('latest_checkpoint.json', dict(checkpoint=str(self.checkpoint), microsteps=800000, optimizer_updates=100000))
        self.write('completed.json', dict(output=str(self.output), microsteps=800000, optimizer_updates=100000))

    def write(self, name, data):
        (self.run / name).write_text(json.dumps(data))

    def test_complete_final_selected(self):
        from contracts import check_final
        self.assertEqual(check_final(self.run, self.output), self.checkpoint)

    def test_missing_completion_blocks(self):
        from contracts import check_final
        (self.run / 'completed.json').unlink()
        with self.assertRaisesRegex(ValueError, 'completed'):
            check_final(self.run, self.output)

    def test_intermediate_or_wrong_round_rejected(self):
        from contracts import check_final
        for delta in [dict(microsteps=760000, optimizer_updates=95000), dict(checkpoint='/tmp/old/800000/pretrained_model')]:
            marker = dict(checkpoint=str(self.checkpoint), microsteps=800000, optimizer_updates=100000)
            marker.update(delta)
            self.write('latest_checkpoint.json', marker)
            with self.assertRaises(ValueError):
                check_final(self.run, self.output)

    def test_missing_processor_state_rejected(self):
        from contracts import check_final
        (self.checkpoint / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'state|missing'):
            check_final(self.run, self.output)

    def test_wrong_action_contract_rejected(self):
        from contracts import check_final
        config = json.loads((self.checkpoint / 'config.json').read_text())
        config['use_relative_actions'] = True
        (self.checkpoint / 'config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'config'):
            check_final(self.run, self.output)

    def test_saved_processor_behavior_must_match_preparation(self):
        from contracts import check_final
        path = self.checkpoint / 'policy_preprocessor.json'
        processor = json.loads(path.read_text())
        for step in processor['steps']:
            if step['registry_name'] == 'relative_actions_processor':
                step['config']['enabled'] = True
        path.write_text(json.dumps(processor))
        with self.assertRaisesRegex(ValueError, 'processor'):
            check_final(self.run, self.output)

    def test_bridge_evidence_rejects_replaced_weights(self):
        from contracts import check_bridge_verification, checkpoint_fingerprints, read, file_identity
        evidence = self.root / 'bridge.json'
        evidence.write_text(json.dumps(dict(passed=True, checkpoint=str(self.checkpoint),
            pins=read(HERE / 'source_snapshot.json'), checkpoint_files=checkpoint_fingerprints(self.checkpoint),
            base_file_identity=file_identity(Path(read(self.checkpoint / 'adapter_config.json')['base_model_name_or_path']) / 'model.safetensors'))))
        check_bridge_verification(self.checkpoint, evidence)
        (self.checkpoint / 'adapter_model.safetensors').write_bytes(b'replaced')
        with self.assertRaisesRegex(ValueError, 'changed|fingerprint'):
            check_bridge_verification(self.checkpoint, evidence)

    def test_run_requires_matching_reload(self):
        from contracts import check_final
        with self.assertRaisesRegex(ValueError, 'reload'):
            check_final(self.run, self.output, require_reload=True)
        self.write('final_checkpoint_reload_verified.json', dict(passed=True, checkpoint='/tmp/other'))
        with self.assertRaisesRegex(ValueError, 'reload'):
            check_final(self.run, self.output, require_reload=True)

    def test_default_is_check_and_all_targets_resolve(self):
        from contracts import parse_args, rollout_arguments
        args = parse_args([])
        self.assertEqual(args.mode, 'check')
        self.assertEqual(args.objects, ['apple', 'peach', 'cup', 'beverage_bottle'])
        args = parse_args(['--dry-run', '--objects', 'bottle', '--episodes-per-object', '2', '--eval-seeds', '123', '456'])
        self.assertEqual(args.objects, ['beverage_bottle'])
        cmd = rollout_arguments(args, self.checkpoint)
        for flag, value in [('--rate', '10'), ('--execution-rate', '60'), ('--execute-steps', '15'),
                            ('--gripper-action-encoding', 'binary'), ('--gripper-min-close-hold-s', '0')]:
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn('--no-freeze-target-until-close', cmd)

    def test_invalid_counts_seeds_and_unknown_flags_fail(self):
        from contracts import parse_args
        for argv in [['--episodes-per-object', '0'], ['--eval-seeds', '1', '1', '--episodes-per-object', '2'],
                     ['--eval-seeds', '-1'], ['--objects', 'mouse'], ['--checkpoint', '/tmp/old']]:
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_training_seeds_and_coverage_come_from_all_records(self):
        from contracts import training_coverage, validate_seeds
        records = []
        for i, obj in enumerate(['apple', 'peach', 'cup', 'beverage_bottle']):
            ep = self.root / f'episode_{i}'
            ep.mkdir()
            (ep / 'episode_summary.json').write_text(json.dumps(dict(scene_randomization=dict(object_positions_m={obj: [.4, -.2, 0]}))))
            records.append(dict(object='bottle' if obj == 'beverage_bottle' else obj, seed=100+i,
                                task=f'Physically grasp the {"bottle" if obj == "beverage_bottle" else obj}, lift it clear, and place it in the box.', source_path=str(ep)))
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps(dict(source_records=records)))
        coverage = training_coverage(manifest, HERE / 'grasp_physics.json')
        self.assertEqual(coverage['training_seeds'], [100, 101, 102, 103])
        self.assertEqual(coverage['objects']['apple']['cells'], [[0, 0]])
        self.assertEqual(coverage['objects']['beverage_bottle']['positions'], [[.4, -.2]])
        with self.assertRaisesRegex(ValueError, 'training'):
            validate_seeds([101], coverage)
        validate_seeds([2026111701], coverage)


if __name__ == '__main__':
    unittest.main()
