import unittest,numpy as np
from scipy.spatial.transform import Rotation
from forward_core import ForwardGuard,GuardViolation,grasp_offset,ee_target
class Guards(unittest.TestCase):
 def test_rotating_offset_target_places_object_at_requested_position(self):
  r=grasp_offset([.4,0,.1],np.eye(3),[.55,0,.12]);rotation=Rotation.from_euler('z',90,degrees=True).as_matrix();goal=np.array([.32,.14,.18]);p=ee_target(goal,rotation,r)
  np.testing.assert_allclose(p+rotation@r,goal,atol=1e-12)
 def test_rejects_recorded_nine_centimetre_reversal(self):
  start=np.array([.509362,-.064491,.18]);goal=np.array([.32,.14,.18]);g=ForwardGuard(start,goal,np.eye(3),'horizontal')
  g.check(start,np.eye(3))
  with self.assertRaisesRegex(GuardViolation,'reverse'):g.check([.468611,-.228732,.18],np.eye(3))
 def test_reversal_is_measured_from_furthest_progress(self):
  g=ForwardGuard([0,0,.18],[.2,0,.18],np.eye(3),'horizontal');g.check([.1,0,.18],np.eye(3));g.check([.098,0,.18],np.eye(3))
  with self.assertRaisesRegex(GuardViolation,'reverse'):g.check([.094,0,.18],np.eye(3))
 def test_forward_lateral_detour_and_small_jitter_allowed(self):
  g=ForwardGuard([0,0,.18],[.2,0,.18],np.eye(3),'horizontal')
  for p in [[0,0,.18],[.05,.01,.18],[.049,.01,.18],[.2,0,.18]]:g.check(p,np.eye(3))
 def test_lift_accepts_under_5mm_but_rejects_over_5mm_sideways(self):
  g=ForwardGuard([.5,-.1,0],[.5,-.1,.18],np.eye(3),'lift')
  g.check([.5,-.1,.1],np.eye(3))
  g.check([.5049,-.1,.12],np.eye(3))
  with self.assertRaises(GuardViolation):g.check([.5051,-.1,.14],np.eye(3))
 def test_attitude_and_nonfinite_inputs_rejected(self):
  for p,r in [([.05,0,.18],Rotation.from_euler('z',.04).as_matrix()),([float('nan'),0,.18],np.eye(3))]:
   g=ForwardGuard([0,0,.18],[.2,0,.18],np.eye(3),'horizontal')
   with self.assertRaises(GuardViolation):g.check(p,r)
 def test_combined_carry_allows_rising_forward_motion_and_wrist_adjustment(self):
  g=ForwardGuard([.5,0,0],[.32,.14,.18],np.eye(3),'carry')
  g.check([.45,.04,.08],Rotation.from_euler('y',.2).as_matrix())
  with self.assertRaisesRegex(GuardViolation,'downward'):g.check([.43,.06,.07],np.eye(3))
 def test_combined_carry_rejects_motion_away_from_box(self):
  g=ForwardGuard([.5,0,0],[.32,.14,.18],np.eye(3),'carry')
  g.check([.45,.04,.08],np.eye(3))
  with self.assertRaisesRegex(GuardViolation,'away'):g.check([.48,.02,.10],np.eye(3))
 def test_carry_can_rise_above_nominal_goal_within_safe_range(self):
  g=ForwardGuard([.5,0,0],[.32,.14,.18],np.eye(3),'carry')
  g.check([.4,.08,.25],np.eye(3))
  with self.assertRaisesRegex(GuardViolation,'safe height'):g.check([.35,.12,.301],np.eye(3))
if __name__=='__main__':unittest.main()
