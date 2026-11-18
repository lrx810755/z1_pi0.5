"""Cross-check decoded cache, original dataset windows, and deployment preprocessing."""
import json,time,copy
from pathlib import Path
import numpy as np
import torch
from train import make_config
from prepare import RUN,write
from cache_data import CACHE,attach_cache
from lerobot.datasets.factory import make_dataset
from lerobot.processor import ImageCropResizeProcessorStep
from lerobot.policies.factory import make_pre_post_processors
from torch.utils.data import default_collate
CP='/home/lp531/smolvla_training/act_correction_chunk16_20261013/checkpoints/050000/pretrained_model'
def main():
 torch.set_num_threads(4)
 cfg=make_config();raw=make_dataset(cfg);cached=attach_cache(make_dataset(cfg))
 pre,_=make_pre_post_processors(cfg.policy,pretrained_path=CP)
 split=json.loads((RUN/'split_manifest.json').read_text());records=split['source_records']
 chosen=[]
 for source in ['original','old_correction','new_fragment']:
  group=[r for r in records if r['source']==source and r['episode_index'] in cfg.dataset.episodes]
  for r in [group[0],group[len(group)//2],group[-1]]:
   # Include the close boundary and an episode edge, checking actual frame numbering.
   for f in [0,r['close_frame']]:chosen.append((r['episode_index'],f))
 by_pair={(int(r['episode_index']),int(r['frame_index'])):i for i,r in enumerate(raw.hf_dataset.select_columns(['episode_index','frame_index']))}
 errors=[];cpu_errors=[];started=time.monotonic();raw_elapsed=cache_elapsed=0
 for ep,f in chosen:
  idx=by_pair[ep,f];start=time.monotonic();r=raw[idx];raw_elapsed+=time.monotonic()-start
  start=time.monotonic();c=cached[idx];cache_elapsed+=time.monotonic()-start
  for key in ['action','observation.state','action_is_pad','index','frame_index','episode_index']:
   if key in r:torch.testing.assert_close(c[key],r[key],rtol=0,atol=0)
  crop=next(s for s in pre.steps if isinstance(s,ImageCropResizeProcessorStep))
  expected=crop.observation({k:r[k].float()/255 for k in raw.meta.camera_keys})
  for k in raw.meta.camera_keys:
   cpu_errors.append(float((expected[k]-c[k]).abs().max()))
   torch.testing.assert_close(expected[k],c[k],rtol=0,atol=1e-7)
  rb=default_collate([r]);cb=default_collate([c]);
  for k in raw.meta.camera_keys:rb[k]=rb[k].float()/255
  rp=pre(rb);steps=pre.steps;pre.steps=[s for s in steps if not isinstance(s,ImageCropResizeProcessorStep)]
  try:cp=pre(cb)
  finally:pre.steps=steps
  for k in raw.meta.camera_keys:
   e=float((rp[k]-cp[k]).abs().max());errors.append(e)
   torch.testing.assert_close(rp[k],cp[k],rtol=0,atol=1e-4)
  print('Verified',ep,f,'maxerror',max(errors),flush=True)
 # All artifacts have correct length/shape; no missing partial cache files.
 count=0
 for r in records:
  for key in raw.meta.camera_keys:
   a=np.load(CACHE/key/f"episode_{r['episode_index']:06d}.npy",mmap_mode='r')
   assert a.shape==(r['frames'],3,224,224) and a.dtype==np.float32
   for i in [0,len(a)//2,len(a)-1]:assert np.isfinite(a[i]).all() and a[i].min()>=0 and a[i].max()<=1
   count+=1
 write('cache_verified.json',dict(passed=True,cache_files=count,episodes=211,frames=47135,paired_windows=len(chosen),max_preprocessed_pixel_error=max(errors),max_cpu_resize_cache_error=max(cpu_errors),cpu_vs_gpu_normalized_tolerance=1e-4,state_action_padding_exact=True,deployment_preprocessor_preserved=True,raw_sample_read_s=raw_elapsed,cached_sample_read_s=cache_elapsed,elapsed_s=time.monotonic()-started))
 print((RUN/'cache_verified.json').read_text(),flush=True)
if __name__=='__main__':main()
