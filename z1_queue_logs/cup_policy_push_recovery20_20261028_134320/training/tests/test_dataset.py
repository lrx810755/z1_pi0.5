"""Catch windows that cross a policy frame or consume another episode."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import cache_data
from rules import valid_anchors

class DatasetBoundaries(unittest.TestCase):
 def test_only_whole_expert_horizon_is_read_and_actions_stop_at30(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);cache=root/'cache';cache.mkdir();n=31
   record=dict(episode_index=0,frames=n,source_path=temp,task='cup',dataset_from_index=0,dataset_to_index=n)
   (root/'split_manifest.json').write_text(json.dumps(dict(source_records=[record])))
   (root/'train_stats.json').write_text('{}');(cache/'completed.json').write_text('{}')
   actions=np.zeros((n,7),np.float32);actions[:,0]=np.arange(n)
   np.savez(cache/'lowdim.npz',actions=actions,states=np.zeros((n,7),np.float32),episode=np.zeros(n,int),frame=np.arange(n))
   expert=np.ones(n,bool);expert[-1]=False
   np.savez(root/'sampling_arrays.npz',eligible_train=valid_anchors(expert),eligible_validation=np.zeros(n,bool))
   for cam in cache_data.CAMERAS:
    (cache/cam).mkdir();np.save(cache/cam/'episode_000000.npy',np.zeros((n,3,224,224),np.float32))
   original=(cache_data.RUN,cache_data.DATA);cache_data.RUN=root;cache_data.DATA=cache
   try:
    dataset=cache_data.CupDataset([0]);sample=dataset[0]
    self.assertEqual(tuple(sample['action'].shape),(30,7));self.assertEqual(sample['action'][-1,0].item(),29)
    self.assertFalse(sample['action_is_pad'].any())
    with self.assertRaises(ValueError):dataset[1]
    with self.assertRaises(ValueError):dataset[30]
   finally:cache_data.RUN,cache_data.DATA=original
