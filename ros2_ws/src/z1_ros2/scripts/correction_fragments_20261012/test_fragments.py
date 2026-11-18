import unittest
import fragment_core as c
class Fragments(unittest.TestCase):
 def test_no_lift_or_contact_cannot_be_accepted(self):
  s=dict(object_position_m=[.4,0,.06],physical_grasp_ok=True,lifted=True,bilateral_contact=True,frozen=False,target_collisions_enabled=True,object_touching_table=False,left_finger_touching_table=False,right_finger_touching_table=False)
  self.assertTrue(c.valid_hold(s))
  for key,value in [('object_position_m',[.4,0,.03]),('physical_grasp_ok',False),('lifted',False),('bilateral_contact',False),('frozen',True),('target_collisions_enabled',False),('object_touching_table',True),('left_finger_touching_table',True)]:
   self.assertFalse(c.valid_hold(dict(s,**{key:value})),key)
 def test_initial_approach_targets_retreat_corridor(self):
  from preflight import routes
  task=dict(kind='left',offset_approach_m=[0,.015,0],rotation_tool_deg=[0,0,0])
  r=next(routes([.3,0,.1],[0,0,0,1],[.5,0,0],task))
  self.assertEqual(r['preparation'][0]['position'],r['preparation'][1]['position'])
  self.assertLess(r['preparation'][0]['position'][0],.3)
 def test_partial_labels_have_no_fabricated_release(self):
  rows=[dict(phase='READY',action=[0]*6+[-.9]) for _ in range(20)]+[dict(phase='GRASPING',action=[0]*6+[-.85]) for _ in range(30)]
  labels,close,valid=c.partial_labels(rows)
  self.assertEqual(close,20);self.assertEqual(labels,[0]*20+[1]*30);self.assertEqual(valid,36)
if __name__=='__main__':unittest.main()
