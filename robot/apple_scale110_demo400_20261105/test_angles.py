import unittest,random
from types import SimpleNamespace as N
from z1_isaac_bridge.grasp_assistant import GraspAssistant
class Angles(unittest.TestCase):
 def node(self):return N(_rng=random.Random(13),azimuth_spread=0.,standoff_min=.27,standoff_max=.3,minimum_pregrasp_z=.05)
 def test_extra_range(self):
  for i in range(16):
   c=GraspAssistant._sample_candidate(self.node(),(.5,0,0),'apple',i,elevation_range_override=(60.,70.))
   self.assertGreaterEqual(c['elevation_deg'],60.);self.assertLessEqual(c['elevation_deg'],70.)
 def test_original_range_unchanged(self):
  c=GraspAssistant._sample_candidate(self.node(),(.5,0,0),'apple',0)
  self.assertGreaterEqual(c['elevation_deg'],70.);self.assertLessEqual(c['elevation_deg'],90.)
 def test_bank_order_and_both_roll_branches(self):
  from assistant import AppleAssistant, base
  self.assertEqual(base.GRASP_PROFILES['apple']['candidate_count'],16)
  n=object.__new__(AppleAssistant)
  for k,v in vars(self.node()).items():setattr(n,k,v)
  for i in range(16):
   c=n._sample_candidate((.5,0,0),'apple',i)
   low,high=(70.,90.)
   self.assertGreaterEqual(c['elevation_deg'],low);self.assertLessEqual(c['elevation_deg'],high)
   self.assertEqual(c['roll_deg'],0. if i%16<8 else 90.)
if __name__=='__main__':unittest.main()
