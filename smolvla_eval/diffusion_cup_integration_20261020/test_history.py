import sys,unittest
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge')
from z1_isaac_bridge.recording_sync import RecordingSynchronizer
class HistoryTest(unittest.TestCase):
 def test_uses_sensor_frame_one_tenth_second_before_current(self):
  from z1_isaac_bridge.diffusion_history import previous_observation
  s=RecordingSynchronizer(['wrist','global'],tolerance=.02)
  for t in [1.,1.1,1.2]:
   for c in s.cameras:s.push(c,t,c)
   s.push('state',t,[t]*7)
  current=s.select_observation(1.2);s.commit(current)
  previous=previous_observation(s,current)
  self.assertAlmostEqual(previous['time'],1.1);self.assertTrue(all(abs(v-1.1)<1e-9 for v in previous['state']))
  self.assertEqual(s.used_images,{'wrist':1.2,'global':1.2})
 def test_does_not_silently_use_stale_or_previous_episode_image(self):
  from z1_isaac_bridge.diffusion_history import previous_observation
  s=RecordingSynchronizer(['wrist'],tolerance=.02)
  s.push('wrist',1.2,None);s.push('state',1.2,[0]*7)
  with self.assertRaises(ValueError):previous_observation(s,s.select_observation(1.2))
if __name__=='__main__':unittest.main()
