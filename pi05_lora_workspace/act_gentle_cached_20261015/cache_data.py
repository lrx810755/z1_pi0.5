"""Memory-mapped, pre-resized RGB observations; raw video metadata remains intact."""
import os
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.dataset_reader import DatasetReader
CACHE=Path('/home/lp531/smolvla_data/act_rgb224_float32_cache_20261015')
def cache_frame_indices(timestamps,fps,length):
 values=np.asarray(timestamps,dtype=np.float64);ids=np.rint(values*fps).astype(np.int64)
 if not np.isfinite(values).all() or (ids<0).any() or (ids>=length).any() or np.max(np.abs(values-ids/fps),initial=0)>1e-4:
  raise ValueError('Cache lookup timestamp outside recorded frame grid')
 return ids.tolist()
class CachedReader(DatasetReader):
 def _query_videos(self,query_timestamps,ep_idx):
  if getattr(self,'_cache_pid',None)!=os.getpid():self._cache_maps={};self._cache_pid=os.getpid()
  result={}
  for key,timestamps in query_timestamps.items():
   pair=(ep_idx,key)
   if pair not in self._cache_maps:
    self._cache_maps[pair]=np.load(CACHE/key/f'episode_{ep_idx:06d}.npy',mmap_mode='r')
   a=self._cache_maps[pair];ids=cache_frame_indices(timestamps,self._meta.fps,len(a))
   result[key]=torch.from_numpy(np.array(a[ids],copy=True)).squeeze(0)
  return result

def attach_cache(dataset):
 if not (CACHE/'completed.json').is_file():raise RuntimeError('Cache build has not completed')
 reader=dataset._ensure_reader()
 if type(reader) is not DatasetReader:raise TypeError(type(reader))
 reader.__class__=CachedReader
 return dataset
