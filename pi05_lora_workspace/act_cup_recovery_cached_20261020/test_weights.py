import unittest
import numpy as np
class ThreePools(unittest.TestCase):
 def test_source_mass_and_key_ratio_with_unequal_pool_sizes(self):
  from weights import source_balanced_weights
  pool=np.array([0,0,0,0,1,1,1,2,2,2,2,2]);key=np.array([0,1,0,1,0,1,0,0,1,1,0,1],bool);eligible=np.ones(12,bool)
  eligible[[3,11]]=False
  w=source_balanced_weights(eligible,key,pool)
  for p,target in enumerate([.7,.15,.15]):
   self.assertAlmostEqual(w[pool==p].sum(),target)
   a=w[(pool==p)&eligible&key][0];b=w[(pool==p)&eligible&~key][0];self.assertAlmostEqual(a/b,2.)
  self.assertEqual(w[3],0);self.assertEqual(w[11],0)
 def test_missing_training_pool_is_an_error(self):
  from weights import source_balanced_weights
  with self.assertRaises(ValueError):source_balanced_weights(np.ones(2,bool),np.zeros(2,bool),np.array([0,1]))
if __name__=='__main__':unittest.main()
