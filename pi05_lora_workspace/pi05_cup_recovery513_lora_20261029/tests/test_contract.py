import unittest
import numpy as np

from dataset_prepare import complete_windows, quantile_stats, subset_identities
from config import make_config, peft_overrides


class ContractTests(unittest.TestCase):
    def test_windows_never_cross_nonexpert_or_tail(self):
        expert = np.ones(65, dtype=bool)
        expert[30] = False
        self.assertEqual(np.flatnonzero(complete_windows(expert)).tolist(), [0, 31, 32, 33, 34, 35])

    def test_quantiles_use_only_selected_training_expert_frames(self):
        x = np.array([[0., 0.], [1., 1.], [9999., 9999.]])
        stats = quantile_stats(x, np.array([True, True, False]))
        np.testing.assert_allclose(stats['q01'], [.01, .01])
        np.testing.assert_allclose(stats['q99'], [.99, .99])
        self.assertEqual(stats['count'], [2])

    def test_subset_accepts_current14_without_claiming19_complete(self):
        expected = {str(i): {} for i in range(19)}
        self.assertEqual(len(subset_identities([str(i) for i in range(14)], expected)), 14)
        with self.assertRaises(ValueError):
            subset_identities(['0', '0'], expected)
        with self.assertRaises(ValueError):
            subset_identities(['unknown'], expected)

    def test_budget_is_optimizer_updates_not_microsteps(self):
        cfg = make_config(validate=False)
        self.assertEqual(cfg.steps, 240000)
        self.assertEqual(cfg.accelerator.gradient_accumulation.steps, 8)
        self.assertEqual(cfg.batch_size, 1)
        self.assertEqual(cfg.policy.chunk_size, 30)
        self.assertEqual(cfg.policy.scheduler_warmup_steps, 8000)
        self.assertEqual(cfg.save_freq, 8000)
        self.assertEqual(cfg.peft.r, 32)
        self.assertEqual(peft_overrides(cfg)['lora_dropout'], 0.)

    def test_real_data_sampler_windows_and_resume(self):
        from dataset import CupDataset
        from sampling import BalancedSampler
        from config import RUN
        import json
        split = json.loads((RUN/'split_manifest.json').read_text())
        data = CupDataset(split['train_episodes'])
        sampler = BalancedSampler(data.meta.episodes['dataset_from_index'],
            data.meta.episodes['dataset_to_index'], episode_indices_to_use=data.episodes,
            absolute_to_relative_idx=data.absolute_to_relative_idx, seed=20261029, shuffle=True)
        order = list(iter(sampler))
        sampler.load_state_dict({'epoch': 0, 'start_index': 123})
        self.assertEqual(list(iter(sampler)), order[123:])
        with np.load(RUN/'sampling_arrays.npz') as a:
            idx = sampler.absolute_indices
            np.testing.assert_allclose(sampler.probabilities.numpy()[a['source_pool'][idx] == 0].sum(), .75)
            np.testing.assert_allclose(sampler.probabilities.numpy()[a['source_pool'][idx] == 1].sum(), .25)
        for ep in [0, 400, 500]:
            ids = [int(i) for i in sampler.absolute_indices if data.arrays['episode'][i] == ep]
            if not ids:
                continue
            absolute = ids[len(ids)//2]
            item = data[data.absolute_to_relative_idx[absolute]]
            self.assertEqual(tuple(item['action'].shape), (30, 7))
            self.assertFalse(item['action_is_pad'].any())
            np.testing.assert_array_equal(item['action'].numpy(), data.arrays['actions'][absolute:absolute+30])
            for camera in data.meta.camera_keys:
                self.assertEqual(tuple(item[camera].shape), (3, 224, 224))
                self.assertGreater(float(item[camera].std()), 0.)
        with self.assertRaises(ValueError):
            # Episode tail must not be usable as an expert action chunk.
            data[data.absolute_to_relative_idx[data.records[0]['dataset_to_index']-1]]


if __name__ == '__main__':
    unittest.main()
