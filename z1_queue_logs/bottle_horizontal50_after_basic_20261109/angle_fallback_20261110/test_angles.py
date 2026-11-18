import sys,unittest
sys.path.insert(0,'/home/lp531/robot/bottle_horizontal_supplement_20261109')
import horizontal50 as h
class Angles(unittest.TestCase):
 def test_original_candidates_first_then_six_relative_fallbacks(self):
  p=h.base.GRASP_PROFILES['beverage_bottle']
  actual=[h.base._candidate_azimuth_deg(-30.,i,p['candidate_azimuth_offsets_deg'],p.get('extra_candidate_azimuths_deg',()),p.get('fallback_relative_azimuths_deg',())) for i in range(p['candidate_count'])]
  self.assertEqual(actual,[-30.,-30.,0.,15.,15.,-60.,-75.,-75.,70.,75.,80.,-20.,-10.,10.,-40.,-50.,-70.])
if __name__=='__main__':unittest.main()
