import unittest
from test_preplanned_transport import Rig
from combined_transport import CombinedBottleTransport
class CombinedRig(CombinedBottleTransport,Rig):pass
class Combined(unittest.TestCase):
 def test_diagonal_carry_is_preplanned_as_one_motion(self):
  n=CombinedRig();n.run_carry();self.assertEqual(n.events,['plan','send'])
  self.assertFalse(n._plan_candidate.call_args.kwargs['linear'])
 def test_unreachable_carry_never_sends_arm_motion(self):
  n=CombinedRig();n._plan_candidate.side_effect=RuntimeError('no IK')
  with self.assertRaisesRegex(RuntimeError,'尚未执行'):n.run_carry()
  n._send_arm_trajectory.assert_not_called()
 def test_continuous_orientation_ranges_replace_thirteen_fixed_candidates(self):
  n=CombinedRig();n._plan_candidate.side_effect=RuntimeError('no IK')
  with self.assertRaises(RuntimeError):n.run_carry()
  self.assertEqual([c.kwargs['orientation_tolerance'] for c in n._plan_candidate.call_args_list],[.2,.4])
 def test_position_constraint_tracks_grasp_offset_as_wrist_rotates(self):
  import assistant
  from types import SimpleNamespace as N
  n=N(base_frame='link00',ee_link='link06',position_tolerance=.001,orientation_tolerance=.2)
  c=assistant.HorizontalAssistant._pose_constraints(n,{'position':[.32,.14,.18],'orientation':[0.,0.,0.,1.],'target_point_offset':[.17,.01,.02]})
  p=c.position_constraints[0]
  self.assertAlmostEqual(p.target_point_offset.x,.17);self.assertAlmostEqual(p.target_point_offset.y,.01);self.assertAlmostEqual(p.target_point_offset.z,.02)
 def test_higher_safe_endpoint_is_accepted_and_executed(self):
  n=CombinedRig();plan=n.plan
  def choose_height(candidate,**kwargs):
   candidate=dict(candidate);candidate['position']=[.32,.14,.25];return plan(candidate,**kwargs)
  n._plan_candidate.side_effect=choose_height;n.run_carry();self.assertAlmostEqual(n.position[2],.25)
 def test_height_range_is_encoded_as_target_region(self):
  import assistant
  from types import SimpleNamespace as N
  n=N(base_frame='link00',ee_link='link06',position_tolerance=.001,orientation_tolerance=.2)
  c=assistant.HorizontalAssistant._pose_constraints(n,{'position':[.32,.14,.18],'orientation':[0.,0.,0.,1.],'position_z_range':[.18,.30]})
  region=c.position_constraints[0].constraint_region
  self.assertAlmostEqual(region.primitive_poses[0].position.z,.24)
  self.assertAlmostEqual(region.primitives[0].dimensions[2],.12)
