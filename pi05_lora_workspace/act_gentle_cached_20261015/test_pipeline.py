import unittest,tempfile
from pathlib import Path
import numpy as np
from cache_data import cache_frame_indices
from sampling import weighted_probabilities
class PipelineTests(unittest.TestCase):
 def test_two_source_masses_double_key_and_exclude_holdout(self):
  ep=np.array([0,0,1,160,160,161]);eligible=np.array([1,1,0,1,1,0],bool);key=np.array([0,1,1,0,1,1],bool)
  p=weighted_probabilities(ep,eligible,key)
  self.assertAlmostEqual(p[ep<160].sum(),.7);self.assertAlmostEqual(p[ep>=160].sum(),.3)
  self.assertAlmostEqual(p[1]/p[0],2);self.assertAlmostEqual(p[4]/p[3],2)
  np.testing.assert_array_equal(p[[2,5]],[0,0])
 def test_cache_timestamp_mapping_uses_local_episode_frames(self):
  self.assertEqual(cache_frame_indices([0.,.1,.200000003],10,12),[0,1,2])
 def test_cache_mapping_rejects_invalid_timestamps(self):
  for times in [[-.1],[1.2],[.05]]:
   with self.assertRaises(ValueError):cache_frame_indices(times,10,12)
if __name__=='__main__':unittest.main()
