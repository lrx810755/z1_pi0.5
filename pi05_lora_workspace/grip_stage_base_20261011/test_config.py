import unittest
from train import make_config, SOURCE, OUTPUT, RUN

class MainExperimentConfigTests(unittest.TestCase):
    def test_fresh_base_and_training_budget(self):
        cfg = make_config()
        self.assertEqual(cfg.policy.pretrained_path, SOURCE)
        self.assertFalse(cfg.policy.use_peft)
        self.assertFalse(cfg.resume)
        self.assertEqual(cfg.steps, 120000)
        self.assertEqual(cfg.accelerator.gradient_accumulation.steps, 8)
        self.assertEqual(cfg.batch_size, 1)
        self.assertEqual(cfg.steps // 8, 15000)
        self.assertEqual(cfg.policy.optimizer_lr, 5e-5)
        self.assertEqual(cfg.policy.scheduler_warmup_steps, 4000)
        self.assertEqual(cfg.policy.scheduler_decay_steps, 120000)
        self.assertEqual(cfg.policy.scheduler_decay_lr, 5e-6)
        self.assertEqual(cfg.save_freq, 4000)
        self.assertEqual(cfg.eval_steps, 4000)
        self.assertEqual(cfg.seed, 20261010)
        self.assertEqual(cfg.policy.chunk_size, 15)
        self.assertEqual(len(cfg.dataset.episodes), 188)
        self.assertIn('grip_stage_base_20261011', str(OUTPUT))
        self.assertNotIn('gentle_correction_base_20261010', str(OUTPUT))
        self.assertTrue((RUN / 'sampling_arrays.npz').is_file())

if __name__ == '__main__':
    unittest.main()
