import unittest
from train import make_config,SOURCE
class BaseConfigTests(unittest.TestCase):
 def test_base_training_contract(self):
  c=make_config()
  self.assertEqual(c.policy.pretrained_path,SOURCE)
  self.assertFalse(c.policy.use_peft) # Load base, then wrap fresh adapters.
  self.assertFalse(c.resume)
  self.assertTrue((SOURCE/'model.safetensors').is_file())
  self.assertFalse((SOURCE/'adapter_model.safetensors').exists())
  self.assertEqual(c.steps,80000)
  self.assertEqual(c.accelerator.gradient_accumulation.steps,8)
  self.assertEqual(c.steps//8,10000)
  self.assertEqual(c.policy.optimizer_lr,5e-5)
  self.assertEqual(c.policy.scheduler_warmup_steps//8,500)
  self.assertEqual(c.save_freq,c.eval_steps)
  self.assertEqual(c.save_freq//8,500)
  self.assertEqual(len(c.dataset.episodes),188)
if __name__=='__main__':unittest.main()
