import unittest
import numpy as np
from sampling import probabilities
class SamplingTests(unittest.TestCase):
 def test_source_ratio_survives_different_phase_composition(self):
  c=np.array([False]*100+[True]*8);k=np.array([True]*20+[False]*80+[True]*7+[False])
  p=probabilities(c,k)
  self.assertAlmostEqual(p[~c].sum(),.7);self.assertAlmostEqual(p[c].sum(),.3)
  self.assertAlmostEqual(p[0]/p[30],2);self.assertAlmostEqual(p[100]/p[-1],2)
 def test_missing_pool_fails(self):
  with self.assertRaises(ValueError):probabilities([False,False],[True,False])
if __name__=='__main__':unittest.main()
