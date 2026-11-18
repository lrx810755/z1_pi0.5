import unittest
from startup import ready_status,read_parameter
from unittest.mock import patch
from types import SimpleNamespace as N
class StartupTests(unittest.TestCase):
 def test_randomizer_alone_not_ready(self):self.assertFalse(ready_status(dict(connected=True,episode_reset=dict(ready=True))))
 def test_all_components_required(self):self.assertTrue(ready_status(dict(connected=True,**{k:dict(ready=True) for k in ['recording','episode_reset','collision_scene','object_collisions']})))
 def test_node_discovery_retry(self):
  with patch('startup.subprocess.run',side_effect=[N(returncode=1,stdout='',stderr='Node not found'),N(returncode=0,stdout='/data\n',stderr='')]) as run,patch('startup.time.sleep'):
   self.assertEqual(read_parameter('dataset_root',attempts=2),'/data');self.assertEqual(run.call_count,2)
 def test_no_fake_ready_on_empty(self):
  with patch('startup.subprocess.run',return_value=N(returncode=0,stdout='',stderr='')),patch('startup.time.sleep'):
   with self.assertRaises(RuntimeError):read_parameter('dataset_root',attempts=2)
if __name__=='__main__':unittest.main()
