import unittest
import json
import tempfile
from pathlib import Path

R = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_cup_pi05_parser_and_execution_window(self):
        from eval_entry import rollout
        args = rollout.parse_args(['--policy-type', 'pi05', '--checkpoint',
            '/home/lp531/smolvla_training/pi05_cup_recovery513_lora_20261029/checkpoints/240000/pretrained_model',
            '--objects', 'cup', '--execute-steps', '12', '--episodes-per-object', '30',
            '--rate', '10', '--execution-rate', '60'])
        self.assertEqual(args.policy_type, 'pi05')
        self.assertEqual(args.execute_steps, 12)
        self.assertEqual(args.episodes_per_object, 30)
        import numpy as np
        self.assertEqual(rollout.execution_window(np.zeros((1, 30, 7)), args.execute_steps).shape, (12, 7))

    def test_all_source_seeds_are_excluded(self):
        excluded = json.loads((R/'excluded_seed_baseline.json').read_text())
        manifest = Path('/home/lp531/pi05_lora_workspace/pi05_cup_recovery513_lora_20261029/split_manifest.json')
        source = json.loads(manifest.read_text())['source_records']
        self.assertEqual(len(excluded), len(set(excluded)))
        self.assertTrue({r['seed'] for r in source} <= set(excluded))

    def test_history_includes_nested_and_interrupted_trial_seeds(self):
        from prepare_runtime import historical_seeds
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rel, seed in [('older/episodes/trial/result.json', 17),
                              ('interrupted/batch/episodes/trial/result.json', 23),
                              ('current/episodes/trial/result.json', 42)]:
                p = root/rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({'seed': seed, 'target_object': 'cup'}))
            nontrial = root/'nontrial/result.json'
            nontrial.parent.mkdir(parents=True)
            nontrial.write_text('[]')
            self.assertEqual(historical_seeds(root, root/'current'), {17, 23})


if __name__ == '__main__':
    unittest.main()
