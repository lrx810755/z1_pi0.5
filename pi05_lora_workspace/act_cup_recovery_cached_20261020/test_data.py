import unittest
from prepare import full_labels
class Labels(unittest.TestCase):
 def test_false_hold_jump_then_open_is_not_close(self):
  a=[-.9]*3+[-.886,-.89,-.9]+[-.9]*3+[-.87,-.8,-.7,-.3,-.3,-.4,-.6,-.8,-.9,-.9]
  rows=[{'action':[0]*6+[x],'phase':'READY'} for x in a]
  labels,close,release=full_labels(rows)
  self.assertEqual(close,9);self.assertEqual(release,14)
  self.assertEqual(labels.tolist(),[0]*9+[1]*5+[0]*5)
if __name__=='__main__':unittest.main()

class DatasetWindows(unittest.TestCase):
 def test_tail_padding_stays_in_episode_and_eligible_window_is_complete(self):
  import json,numpy as np
  from prepare import RUN
  from cache_data import CupDataset
  record=json.loads((RUN/'split_manifest.json').read_text())['source_records'][0]
  ds=CupDataset([0],raw_images=True);n=record['frames']
  complete=ds[n-16];tail=ds[n-1]
  self.assertFalse(complete['action_is_pad'].any())
  self.assertEqual(int(tail['action_is_pad'].sum()),15)
  np.testing.assert_array_equal(tail['action'][0],tail['action'][-1])
