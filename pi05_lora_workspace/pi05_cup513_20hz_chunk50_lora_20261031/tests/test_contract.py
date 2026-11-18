import unittest,json
import numpy as np
from config import RUN,make_config,peft_overrides
class ContractTests(unittest.TestCase):
 def test_budget_and_horizon(self):
  c=make_config(validate=False);self.assertEqual(c.policy.chunk_size,50);self.assertEqual(c.steps,240000);self.assertEqual(c.batch_size,1);self.assertEqual(c.accelerator.gradient_accumulation.steps,8);self.assertEqual(c.policy.optimizer_lr,5e-5);self.assertEqual(c.policy.scheduler_warmup_steps,8000);self.assertEqual(peft_overrides(c)['lora_dropout'],0.)
 def test_sampling_excludes_nonexpert_and_horizon_tails(self):
  from dataset import CupDataset
  from sampling import BalancedSampler
  d=json.loads((RUN/'split_manifest.json').read_text());data=CupDataset(d['train_episodes']);self.assertEqual(data.fps,20)
  sampler=BalancedSampler(data.meta.episodes['dataset_from_index'],data.meta.episodes['dataset_to_index'],episode_indices_to_use=data.episodes,absolute_to_relative_idx=data.absolute_to_relative_idx,seed=20261029,shuffle=True)
  with np.load(RUN/'sampling_arrays.npz') as a:
   ids=sampler.absolute_indices;self.assertTrue(a['eligible_train'][ids].all());self.assertFalse(a['eligible_validation'][ids].any())
   np.testing.assert_allclose(sampler.probabilities.numpy()[a['source_pool'][ids]==0].sum(),.75)
   np.testing.assert_allclose(sampler.probabilities.numpy()[a['source_pool'][ids]==1].sum(),.25)
   for ep in [0,400,500]:
    valid=np.flatnonzero(a['eligible_train']&(a['episode']==ep))
    if not len(valid):continue
    i=int(valid[len(valid)//2]);x=data[data.absolute_to_relative_idx[i]];self.assertEqual(tuple(x['action'].shape),(50,7));self.assertFalse(x['action_is_pad'].any());self.assertTrue(a['expert'][i:i+50].all())
    for cam in data.meta.camera_keys:self.assertEqual(tuple(x[cam].shape),(3,224,224));self.assertGreater(float(x[cam].std()),0)
  with self.assertRaises(ValueError):data[data.absolute_to_relative_idx[data.records[0]['dataset_to_index']-1]]
 def test_sampler_resume_is_reproducible(self):
  from dataset import CupDataset
  from sampling import BalancedSampler
  d=json.loads((RUN/'split_manifest.json').read_text());data=CupDataset(d['train_episodes']);s=BalancedSampler(data.meta.episodes['dataset_from_index'],data.meta.episodes['dataset_to_index'],episode_indices_to_use=data.episodes,absolute_to_relative_idx=data.absolute_to_relative_idx,seed=20261029,shuffle=True)
  order=list(iter(s));s.load_state_dict({'epoch':0,'start_index':123});self.assertEqual(list(iter(s)),order[123:])
