import re
import unittest
import train

class ConfigTest(unittest.TestCase):
    def test_binary_dataset_and_requested_training_scope(self):
        cfg = train.make_config()
        self.assertEqual(cfg.policy.chunk_size, 15)
        self.assertEqual(cfg.policy.optimizer_lr, 5e-5)
        self.assertEqual(cfg.dataset.eval_split, 0)
        self.assertEqual(cfg.eval_steps, 0)
        self.assertEqual(cfg.batch_size * cfg.accelerator.gradient_accumulation.steps, 8)
        self.assertIn('binary_20261007', str(cfg.dataset.root))
        for network in ['paligemma.model.language_model', 'gemma_expert.model']:
            for kind, suffixes in [('self_attn', ['q_proj','k_proj','v_proj','o_proj']), ('mlp', ['gate_proj','up_proj','down_proj'])]:
                for suffix in suffixes:
                    name = f'model.paligemma_with_expert.{network}.layers.0.{kind}.{suffix}'
                    self.assertIsNotNone(re.fullmatch(cfg.peft.target_modules, name), name)
        self.assertIsNone(re.fullmatch(cfg.peft.target_modules, 'model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj'))

if __name__ == '__main__': unittest.main()
