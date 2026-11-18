"""All sources decoded from their actual20Hz raw RGB; no10Hz crop cache."""
import json,os,hashlib
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from config import RUN
from dataset_prepare import CAMERAS,CROPS,HORIZON,FPS
class CupDataset(torch.utils.data.Dataset):
 def __init__(self,episodes):
  self.episodes=list(episodes);self.records=json.loads((RUN/'split_manifest.json').read_text())['source_records']
  with np.load(RUN/'lowdim.npz') as a:self.arrays={k:a[k] for k in a.files}
  with np.load(RUN/'sampling_arrays.npz') as a:self.approved=a['eligible_train']|a['eligible_validation']
  with np.load(RUN/'image_pins.npz') as a:self.image_pins={k:a[k] for k in a.files}
  self.indices=np.flatnonzero(np.isin(self.arrays['episode'],self.episodes));self.absolute_to_relative_idx={int(x):i for i,x in enumerate(self.indices)}
  self.num_frames=len(self.indices);self.num_episodes=len(self.episodes);self.root=RUN;self.fps=FPS
  features={k:dict(dtype='float32',shape=[7],names=[f'joint{i}' for i in range(1,7)]+['jointGripper']) for k in ['action','observation.state']}
  for c in CAMERAS:features[c]=dict(dtype='image',shape=[360 if c.endswith('wrist') else 480,640,3],names=['height','width','channels'])
  stats={k:{n:np.asarray(v,dtype=np.float32) for n,v in d.items()} for k,d in json.loads((RUN/'train_stats.json').read_text()).items()}
  self.meta=SimpleNamespace(features=features,stats=stats,camera_keys=CAMERAS,has_language_columns=False,fps=FPS,robot_type='unitree_z1',repo_id='lp531/z1_cup513_20hz_chunk50_pi05',episodes={k:[r[k] for r in self.records] for k in ['dataset_from_index','dataset_to_index']})
 def __len__(self):return self.num_frames
 def __getitem__(self,index):
  absolute=int(self.indices[index]);ep=int(self.arrays['episode'][absolute]);frame=int(self.arrays['frame'][absolute]);r=self.records[ep]
  if not self.approved[absolute] or absolute+HORIZON>r['dataset_to_index']:raise ValueError('Require complete50-step expert window')
  result={'observation.state':torch.from_numpy(self.arrays['states'][absolute].copy()),'action':torch.from_numpy(self.arrays['actions'][absolute:absolute+HORIZON].copy()),'action_is_pad':torch.zeros(HORIZON,dtype=torch.bool),'episode_index':torch.tensor(ep),'frame_index':torch.tensor(frame),'index':torch.tensor(absolute),'timestamp':torch.tensor(frame/FPS),'task':r['task'],'task_index':torch.tensor(0)}
  if getattr(self,'_pid',None)!=os.getpid():self._rows=OrderedDict();self._pid=os.getpid()
  if ep not in self._rows:
   if len(self._rows)>=20:self._rows.popitem(last=False)
   self._rows[ep]=[json.loads(x) for x in (Path(r['source_path'])/'frames.jsonl').read_text().splitlines()]
  self._rows.move_to_end(ep);row=self._rows[ep][frame]
  from PIL import Image
  from torchvision.transforms import functional as F
  for j,c in enumerate(CAMERAS):
   path=Path(r['source_path'])/row[c];st=path.stat();expected=(int(self.image_pins['image_sizes'][absolute,j]),int(self.image_pins['image_mtime_ns'][absolute,j]),int(self.image_pins['image_ctime_ns'][absolute,j]))
   if (st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=expected:
    if hashlib.sha256(path.read_bytes()).hexdigest()!=self.image_pins['image_hashes'][absolute,j].decode():raise ValueError('Source JPEG changed after pinning: '+str(path))
   with Image.open(path) as im:
    expected_size=(640,360 if c.endswith('wrist') else 480)
    if im.size!=expected_size or im.mode!='RGB':raise ValueError('Wrong source RGB image shape/mode')
    pixels=torch.from_numpy(np.array(im)).permute(2,0,1).float()/255
   result[c]=F.resize(F.crop(pixels,*CROPS[c]),[224,224],antialias=True).clamp(0,1)
  return result
