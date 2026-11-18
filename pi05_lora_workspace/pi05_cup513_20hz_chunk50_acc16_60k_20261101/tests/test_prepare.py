import unittest
import numpy as np
from dataset_prepare import complete_windows,quantile_stats
class PreparationTests(unittest.TestCase):
 def test_50step_windows_exclude_intervention_and_tail(self):
  x=np.ones(105,dtype=bool);x[50]=False
  self.assertEqual(np.flatnonzero(complete_windows(x)).tolist(),[0,51,52,53,54,55])
 def test_cannot_use_30step_tail_as50step_window(self):
  self.assertFalse(complete_windows(np.ones(49,dtype=bool)).any())
 def test_stats_exclude_heldout_and_intervention(self):
  s=quantile_stats(np.array([[0.],[1.],[9999.]]),np.array([True,True,False]))
  np.testing.assert_allclose(s['q01'],[.01]);np.testing.assert_allclose(s['q99'],[.99])
if __name__=='__main__':unittest.main()
