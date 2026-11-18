import unittest,numpy as np
from core import Settings,waypoints,validate_timing,route_candidates
class Tests(unittest.TestCase):
 def test_speed(self):self.assertAlmostEqual(Settings().final_speed,.064)
 def test_route(self):
  p=np.array([.4,0,.05]);q=np.array([.5,0,.03]);out=waypoints(p,q,Settings());self.assertAlmostEqual(out[2],p[2]);self.assertAlmostEqual(np.linalg.norm(out-p),.06);self.assertLess(out[0],p[0])
 def test_route_candidates_preserve_final_horizontal_axis(self):
  final=np.array([.4,0,.08]);routes=route_candidates(final,[.55,0,0],[1,0,0],Settings())
  self.assertEqual(len(routes),24)
  for route in routes:
   high,low=route['high'],route['low']
   self.assertAlmostEqual(low[2],final[2]);self.assertGreater(high[2],low[2]);self.assertAlmostEqual(low[1],final[1]);self.assertLess(low[0],final[0])
  self.assertEqual({x['retreat_m'] for x in routes},{.05,.06,.08,.10})
  self.assertEqual({x['azimuth_deg'] for x in routes},{0.,15.,-15.})
 def test_candidates_are_not_duplicates(self):
  routes=route_candidates([.4,0,.08],[.55,0,0],[1,0,0],Settings())
  self.assertEqual(len({tuple(np.r_[x['high'],x['low']]) for x in routes}),24)
 def test_no_stop(self):
  with self.assertRaises(ValueError):validate_timing([0,1,2],[[0]*6,[0]*6,[0]*6])
 def test_monotone(self):
  with self.assertRaises(ValueError):validate_timing([0,1,1],[[0]*6,[.1]*6,[0]*6])
 def test_pass(self):validate_timing([0,1,2],[[0]*6,[.1]*6,[0]*6])
if __name__=='__main__':unittest.main()
