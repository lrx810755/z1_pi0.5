import unittest
import torch


class TrainingGuardTests(unittest.TestCase):
    def test_strict_load_rejects_silent_native_fallback(self):
        from guards import strict_base_load
        class Model(torch.nn.Linear):
            pass
        with self.assertRaisesRegex(RuntimeError, 'strict base'):
            strict_base_load(lambda: Model(2, 2), Model)

    def test_strict_load_rejects_partial_load_even_if_native_swallows_it(self):
        from guards import strict_base_load
        class Model(torch.nn.Linear):
            pass
        def factory():
            policy = Model(2, 2)
            try:
                policy.load_state_dict({'weight': torch.ones(2, 2)}, strict=False)
            except RuntimeError:
                pass  # This is native PI05's silent return path.
            return policy
        with self.assertRaisesRegex(RuntimeError, 'strict base'):
            strict_base_load(factory, Model)

    def test_successful_strict_load_preserves_exact_weights(self):
        from guards import strict_base_load
        class Model(torch.nn.Linear):
            pass
        state = {'weight': torch.ones(2, 2), 'bias': torch.zeros(2)}
        def factory():
            policy = Model(2, 2)
            policy.load_state_dict(state)
            return policy
        policy, report = strict_base_load(factory, Model)
        self.assertEqual(report['loaded_keys'], 2)
        self.assertTrue(torch.equal(policy.weight, state['weight']))
        self.assertNotIn('load_state_dict', Model.__dict__)

    def test_validation_nan_inf_and_missing_value_fail(self):
        from guards import validation_evidence
        for value in ['nan', 'inf', '-inf', '']:
            with self.subTest(value=value), self.assertRaises((ValueError, RuntimeError)):
                validation_evidence('step 8000: eval_loss='+value)
        self.assertIsNone(validation_evidence('normal training log'))
        self.assertEqual(validation_evidence('step 8000: eval_loss=0.1234')['optimizer_updates'], 500)


if __name__ == '__main__':
    unittest.main()
