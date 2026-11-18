import unittest
from basic_nodes import collection_ready
class ReadinessTests(unittest.TestCase):
 def full_status(self):
  return dict(connected=True,recording=dict(ready=True),episode_reset=dict(ready=True),collision_scene=dict(ready=True),object_collisions=dict(ready=True))
 def test_scene_without_assistant_is_not_ready(self):
  self.assertFalse(collection_ready(self.full_status(),'apple'))
 def test_assistant_must_match_requested_target(self):
  s=self.full_status();s['assistant']=dict(target_object='cup',phase='IDLE')
  self.assertFalse(collection_ready(s,'apple'));s['assistant']['target_object']='apple';self.assertTrue(collection_ready(s,'apple'))
if __name__=='__main__':unittest.main()
