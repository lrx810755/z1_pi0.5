import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
class LoggingTests(unittest.TestCase):
 def test_discarded_tail_close_is_logged_without_changing_execution(self):
  spec=importlib.util.spec_from_file_location('chunk_logging',P/'chunk_logging.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  chunk=np.zeros((1,15,7),np.float32);chunk[0,:,0]=np.arange(15);chunk[0,12:,6]=1;original=chunk.copy()
  with tempfile.TemporaryDirectory() as tmp:
   f=Path(tmp)/'predictions.jsonl';actual=mod.record_prediction(chunk,10,f,{'prediction':3});row=json.loads(f.read_text())
   self.assertEqual(len(row['actions']),15);self.assertEqual(row['first_close_index'],12);self.assertEqual(row['executed_prefix_close_indices'],[]);self.assertEqual(row['discarded_tail_close_indices'],[12,13,14]);np.testing.assert_array_equal(actual,chunk[0,:10]);np.testing.assert_array_equal(chunk,original)
 def test_executed_close_and_open_tail_are_distinguished(self):
  spec=importlib.util.spec_from_file_location('chunk_logging',P/'chunk_logging.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  chunk=np.zeros((1,15,7),np.float32);chunk[0,:11,6]=1
  with tempfile.TemporaryDirectory() as tmp:
   f=Path(tmp)/'predictions.jsonl';mod.record_prediction(chunk,10,f,{'prediction':0});mod.record_prediction(chunk,10,f,{'prediction':1});rows=[json.loads(x) for x in f.read_text().splitlines()];self.assertEqual(len(rows),2);self.assertEqual(rows[0]['discarded_tail_open_indices'],[11,12,13,14]);self.assertEqual(rows[1]['prediction'],1)
if __name__=='__main__':unittest.main()
