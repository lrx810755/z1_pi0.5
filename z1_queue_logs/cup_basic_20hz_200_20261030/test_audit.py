import copy,unittest
from audit import audit_rows
class TimingTests(unittest.TestCase):
 def setUp(self):
  self.rows=[]
  for i in range(4):
   t=10+i*.05
   self.rows.append(dict(frame_index=i,timestamp=i*.05,simulation_time=t,joint_state_stamp=t,joint_state_source_stamps=[t-.016,t+.001],wrist_image_stamp=t,global_image_stamp=t,global_right_image_stamp=t,action_stamp=t-.01,action_age_s=.01))
 def test_valid(self):self.assertEqual(audit_rows(self.rows)['frames'],4)
 def test_reused_camera(self):
  self.rows[1]['wrist_image_stamp']=10
  with self.assertRaises(ValueError):audit_rows(self.rows)
 def test_grid_gap(self):
  self.rows[2]['simulation_time']+=.05
  with self.assertRaises(ValueError):audit_rows(self.rows)
 def test_future_command(self):
  self.rows[1]['action_stamp']+=.02
  with self.assertRaises(ValueError):audit_rows(self.rows)
 def test_stale_command(self):
  self.rows[1]['action_stamp']-=.1
  with self.assertRaises(ValueError):audit_rows(self.rows)
 def test_spread(self):
  self.rows[1]['global_right_image_stamp']+=.021
  with self.assertRaises(ValueError):audit_rows(self.rows)
 def test_state_gap(self):
  self.rows[1]['joint_state_source_stamps'][0]-=.1
  with self.assertRaises(ValueError):audit_rows(self.rows)
if __name__=='__main__':unittest.main()
