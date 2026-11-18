"""Torch dataset adapter for immutable raw cup recordings and decoded cache."""
import json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from prepare import RUN,DATA,CAMERAS
from windows import indices as window_indices
CACHE=DATA
class CupDataset(torch.utils.data.Dataset):
 def __init__(self,episodes,raw_images=False):
  self.episodes=list(episodes);self.raw_images=raw_images
  self.records=json.loads((RUN/'split_manifest.json').read_text())['source_records']
  with np.load(DATA/'lowdim.npz') as a:self.arrays={k:a[k] for k in a.files}
  self.indices=np.flatnonzero(np.isin(self.arrays['episode'],self.episodes))
  self.absolute_to_relative_idx={int(x):i for i,x in enumerate(self.indices)}
  self.num_frames=len(self.indices);self.num_episodes=len(self.episodes);self.root=DATA;self.fps=10
  features={k:dict(dtype='float32',shape=[7],names=[f'joint{i}' for i in range(1,7)]+['jointGripper']) for k in ['action','observation.state']}
  for cam in CAMERAS:features[cam]=dict(dtype='image',shape=[224,224,3],names=['height','width','channels'])
  stats={k:{n:np.array(v,dtype=np.float32) for n,v in s.items()} for k,s in json.loads((RUN/'train_stats.json').read_text()).items()}
  self.meta=SimpleNamespace(features=features,stats=stats,camera_keys=CAMERAS,has_language_columns=False,fps=10,robot_type='unitree_z1',repo_id='lp531/z1_cup_recovery_raw_cached_20261020',episodes={k:[r[k] for r in self.records] for k in ['dataset_from_index','dataset_to_index']})
  if not raw_images and not (DATA/'completed.json').exists():raise RuntimeError('Decoded cache not complete')
 def __len__(self):return self.num_frames
 def __getitem__(self,index):
  absolute=int(self.indices[index]);ep=int(self.arrays['episode'][absolute]);frame=int(self.arrays['frame'][absolute]);r=self.records[ep]
  obs_ids,ids,padding=window_indices(r['dataset_from_index'],r['dataset_to_index'],absolute)
  result={'observation.state':torch.from_numpy(self.arrays['states'][obs_ids].copy()),'action':torch.from_numpy(self.arrays['actions'][ids].copy()),'action_is_pad':torch.from_numpy(padding),'episode_index':torch.tensor(ep),'frame_index':torch.tensor(frame),'index':torch.tensor(absolute),'timestamp':torch.tensor(frame/10.),'task':r['task'],'task_index':torch.tensor(0)}
  if getattr(self,'_pid',None)!=os.getpid():self._maps={};self._rows={};self._pid=os.getpid()
  for cam in CAMERAS:
   if self.raw_images:
    from PIL import Image
    root=Path(r['source_path'])
    if ep not in self._rows:self._rows[ep]=[json.loads(l) for l in (root/'frames.jsonl').read_text().splitlines()]
    frames=[]
    for obs_index in obs_ids-r['dataset_from_index']:
     with Image.open(root/self._rows[ep][int(obs_index)][cam]) as im:a=np.array(im.convert('RGB'))
     frames.append(torch.from_numpy(a).permute(2,0,1))
    result[cam]=torch.stack(frames)
   else:
    pair=(ep,cam)
    if pair not in self._maps:self._maps[pair]=np.load(DATA/cam/f'episode_{ep:06d}.npy',mmap_mode='r')
    result[cam]=torch.from_numpy(self._maps[pair][obs_ids-r['dataset_from_index']].copy())
  return result
