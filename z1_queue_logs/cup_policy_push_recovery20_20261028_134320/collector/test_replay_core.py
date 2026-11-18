import unittest
from replay_core import command_prefix, validate_takeover
class Tests(unittest.TestCase):
 def test_prefix_never_executes_future(self):
  self.assertEqual([x['simulation_elapsed_s'] for x in command_prefix([{'simulation_elapsed_s':x} for x in [0,.1,.2,.3]],.2)],[0,.1,.2])
 def test_reject_past_tilt_flight_or_unstable(self):
  good=dict(cylinder_tilt_valid=True,cylinder_tilt_deg=1.,object_touching_table=True,object_speed_m_s=.001,object_position_m=[.5,0,0],frozen=False,target_collisions_enabled=True,gripper_position_rad=-.9,physical_grasp_ok=False)
  self.assertTrue(validate_takeover(good,[.48,0,0]))
  for updates in [dict(cylinder_tilt_deg=20),dict(object_position_m=[.5,0,.2]),dict(object_touching_table=False),dict(object_speed_m_s=.2),dict(frozen=True)]:
   with self.assertRaises(ValueError):validate_takeover(dict(good,**updates),[.48,0,0])
if __name__=='__main__':unittest.main()
