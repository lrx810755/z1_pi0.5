import unittest,json
from pathlib import Path
from collections import Counter
import numpy as np
from config import RUN,ACCUMULATION,make_config
from rules import valid_anchors
class DataContract(unittest.TestCase):
 def test_subset_counts_splits_and_task_targets(self):
  d=json.loads((RUN/'split_manifest.json').read_text());records=d['source_records']
  self.assertEqual(Counter(r['source'] for r in records),dict(apple_full=133,apple_correction=59,cup_basic=200,cup_correction=50))
  self.assertEqual(len(d['train_episodes']),395);self.assertEqual(len(d['validation_episodes']),47)
  self.assertFalse({r['seed'] for r in records if r['split']=='train'}&{r['seed'] for r in records if r['split']=='validation'})
  for r in records:
   self.assertIn(r['object'],r['task']);self.assertEqual('place' in r['task'],r['source']!='apple_correction')
  basic=Counter(tuple(r['grid']) for r in records if r['source']=='cup_basic');self.assertEqual(len(basic),20);self.assertEqual(set(basic.values()),{10})
 def test_windows_and_weight_mass(self):
  d=json.loads((RUN/'split_manifest.json').read_text());a=np.load(RUN/'sampling_arrays.npz');low=np.load(RUN/'lowdim.npz')
  for r in d['source_records']:
   start,end=r['dataset_from_index'],r['dataset_to_index'];self.assertTrue(np.array_equal(a['eligible_train'][start:end]|a['eligible_validation'][start:end],valid_anchors(a['expert'][start:end],30)))
  self.assertTrue(np.isin(low['actions'][:,-1],[0,1]).all())
  self.assertTrue((a['weights'][~a['eligible_train']]==0).all())
  for pool,mass in enumerate([.35,.15,.375,.125]):self.assertAlmostEqual(a['weights'][a['source_pool']==pool].sum(),mass)
 def test_budget(self):
  c=make_config(validate=False);self.assertEqual(ACCUMULATION,16);self.assertEqual(c.steps//ACCUMULATION,10000);self.assertEqual(c.policy.chunk_size,30);self.assertEqual(c.save_freq//ACCUMULATION,1000);self.assertEqual(c.policy.scheduler_warmup_steps//ACCUMULATION,500)
if __name__=='__main__':unittest.main()
