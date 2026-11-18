import unittest
import numpy as np
class AlignmentTests(unittest.TestCase):
 def test_history_and_execution_alignment(self):
  from windows import indices
  obs,act,pad=indices(100,150,110)
  np.testing.assert_array_equal(obs,[109,110])
  np.testing.assert_array_equal(act[1:],np.arange(110,125))
  self.assertEqual(int(act[0]),109);self.assertFalse(pad.any())
 def test_first_observation_cannot_cross_episode(self):
  from windows import indices
  obs,act,pad=indices(100,150,100)
  np.testing.assert_array_equal(obs,[100,100]);self.assertEqual(int(act[0]),100)
  self.assertTrue(pad[0]);self.assertFalse(pad[1:].any())
 def test_tail_padding_is_explicit(self):
  from windows import indices
  obs,act,pad=indices(100,120,119)
  self.assertTrue((act<=119).all());self.assertEqual(int(pad.sum()),14)
if __name__=='__main__':unittest.main()
