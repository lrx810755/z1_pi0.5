"""Verify every cache shape and compare real JPEG preprocessing at event boundaries."""
import json,time
import numpy as np
import torch
from torchvision.transforms import functional as F
from prepare import RUN,DATA,CAMERAS,CROPS,write
from cache_data import CupDataset

def main():
 torch.set_num_threads(2);split=json.loads((RUN/'split_manifest.json').read_text());records=split['source_records'];all_eps=list(range(len(records)))
 raw=CupDataset(all_eps,raw_images=True);cache=CupDataset(all_eps);errors=[];pairs=0
 for r in records:
  for cam in CAMERAS:
   a=np.load(DATA/cam/f'episode_{r["episode_index"]:06d}.npy',mmap_mode='r');assert a.shape==(r['frames'],3,224,224) and a.dtype==np.float32
   for idx in [0,len(a)//2,len(a)-1]:assert np.isfinite(a[idx]).all() and a[idx].min()>=0 and a[idx].max()<=1
 selected=[]
 for source in ['original','old_correction','new_correction']:
  for split_name in ['train','validation']:
   group=[r for r in records if r['source']==source and r['split']==split_name]
   selected.extend([group[0],group[len(group)//2],group[-1]])
 for r in selected:
  for frame in [0,r['close_frame'],r['open_frame'],r['frames']-16]:
   idx=r['dataset_from_index']+frame;x=raw[idx];y=cache[idx]
   for name in ['action','observation.state','action_is_pad']:torch.testing.assert_close(x[name],y[name],rtol=0,atol=0)
   assert not y['action_is_pad'].any()
   for cam in CAMERAS:
    expected=F.resize(F.crop(x[cam].float()/255,*CROPS[cam]),[224,224]).clamp(0,1)
    errors.append(float((expected-y[cam]).abs().max()));torch.testing.assert_close(expected,y[cam],rtol=0,atol=1e-7)
   pairs+=1
 with np.load(RUN/'sampling_arrays.npz') as a:
  assert not np.isin(a['episode'][a['weights']>0],split['validation_episodes']).any()
  for r in records:
   mask=a['episode']==r['episode_index'];valid=a['eligible_train'][mask]|a['eligible_validation'][mask]
   assert np.all(a['frame'][mask][valid]+15<r['frames'])
  for pool,target in enumerate([.7,.15,.15]):assert abs(a['weights'][a['source_pool']==pool].sum()-target)<1e-12
 write('cache_verified.json',dict(passed=True,cache_files=len(records)*3,episodes=len(records),paired_windows=pairs,max_pixel_error=max(errors),raw_action_state_exact=True,train_val_disjoint=True,complete_chunks=True))
 print((RUN/'cache_verified.json').read_text(),flush=True)
if __name__=='__main__':main()
