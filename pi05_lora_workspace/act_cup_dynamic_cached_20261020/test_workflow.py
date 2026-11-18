import json,tempfile,unittest
from pathlib import Path
from workflow_core import collection_ready,classify_error,in_box_released

class CollectionGate(unittest.TestCase):
 def fixture(self,count,quota_ok=True):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
  root=Path(tmp.name);data=root/'data';log=root/'log';data.mkdir();log.mkdir()
  for i in range(count):
   p=data/f'episode_{i:06d}';p.mkdir();cell=i//8
   s=dict(target_object='cup',task_success=True,physical_grasp_ok=True,scene_randomization=dict(target_grid=[cell%5,cell//5]))
   (p/'episode_summary.json').write_text(json.dumps(s))
  (log/'completed.json').write_text(json.dumps(dict(complete=True,exit_code=0,saved=count)))
  (log/'final_verified.json').write_text(json.dumps(dict(quota_verified=quota_ok,recordings_verified=count)))
  return data,log
 def test_never_train_on_159_episodes(self):
  data,log=self.fixture(159);ready,info=collection_ready(data,log);self.assertFalse(ready)
 def test_wait_for_verifier_even_when_counter_says_160(self):
  data,log=self.fixture(160,False);self.assertFalse(collection_ready(data,log)[0])
 def test_exact_twenty_cells_eight_each_is_ready(self):
  data,log=self.fixture(160);self.assertTrue(collection_ready(data,log)[0])
 def test_imbalanced_cells_do_not_pass(self):
  data,log=self.fixture(160);p=data/'episode_000159'/'episode_summary.json';s=json.loads(p.read_text());s['scene_randomization']['target_grid']=[0,0];p.write_text(json.dumps(s));self.assertFalse(collection_ready(data,log)[0])

class EvaluationRules(unittest.TestCase):
 def test_position_alone_is_insufficient(self):
  s=dict(object_position_m=[.32,.14,.05],released=False,holding_observed=False)
  self.assertFalse(in_box_released(s));s['released']=True;self.assertTrue(in_box_released(s));s['holding_observed']=True;self.assertFalse(in_box_released(s))
 def test_only_blank_camera_errors_are_replaced(self):
  self.assertEqual(classify_error(dict(error_detail='blank_camera:wrist',cleanup_error='')),'replace')
  self.assertEqual(classify_error(dict(error_detail='stale_stream',cleanup_error='')),'fatal')
  self.assertEqual(classify_error(dict(error_detail='',cleanup_error='publisher invalid')),'fatal')
  self.assertEqual(classify_error(dict(error_detail='',cleanup_error='')),'valid')
if __name__=='__main__':unittest.main()
