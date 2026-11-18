import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torchvision.transforms import functional as F
from cache_data import CupDataset
from prepare import RUN,DATA,CAMERAS,write
from sampling import BalancedSampler
split=json.loads((RUN/'split_manifest.json').read_text());ds=CupDataset(split['train_episodes']+split['validation_episodes']);raw=CupDataset(ds.episodes,raw_images=True)
a=np.load(RUN/'sampling_arrays.npz');checks=[]
for rec in split['source_records']:
 for frame in [0,1,rec['close_frame']-1]:
  absolute=rec['dataset_from_index']+frame;item=ds[absolute]
  expected=[max(rec['dataset_from_index'],absolute-1),absolute]
  assert np.array_equal(item['observation.state'],ds.arrays['states'][expected])
  assert np.array_equal(item['action'][1:],ds.arrays['actions'][absolute:absolute+15])
  assert np.isin(item['action'][:,-1],[0,1]).all()
for source in ['original','old_correction','new_correction']:
 for subset in ['train','validation']:
  rec=next(x for x in split['source_records'] if x['source']==source and x['split']==subset)
  anchor=rec['dataset_from_index']+max(1,rec['close_frame']-5);cached=ds[anchor];jpg=raw[anchor]
  for cam in CAMERAS:
   top=-140 if cam.endswith('wrist') else -80
   transformed=F.resize(F.crop(jpg[cam].float()/255,top,0,640,640),(224,224)).clamp(0,1)
   error=float((transformed-cached[cam]).abs().max());assert error<=1e-6,error
   checks.append(dict(source=source,split=subset,cam=cam,max_error=error))
assert len(a['weights'][a['eligible_train']])==44723
assert abs(a['weights'].sum()-1)<1e-10
for pool,mass in enumerate([.7,.15,.15]):assert abs(a['weights'][a['source_pool']==pool].sum()-mass)<1e-10
assert not (a['eligible_train']&a['eligible_validation']).any()
write('data_verified.json',dict(episodes=249,all_episode_boundary_and_action_checks=747,raw_cached_history_image_checks=checks,source_mass=[.7,.15,.15],training_windows=44723,sampling_arrays_sha256=hashlib.sha256((RUN/'sampling_arrays.npz').read_bytes()).hexdigest(),cache_reused=str(DATA),no_new_video_decode=True))
print('all alignment, source mass, binary label and RGB cache checks passed')
