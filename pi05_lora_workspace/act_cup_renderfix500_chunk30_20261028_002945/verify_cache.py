"""Audit all cache digests, masks and raw/cached event-adjacent pixel equality."""
import argparse,json,time
import numpy as np
import torch
from torchvision.transforms import functional as F
from prepare import RUN,DATA,CAMERAS,CROPS,write,sha,collection_gate,assert_pins
from cache_data import CupDataset
from rules import valid_anchors

def main():
 p=argparse.ArgumentParser();p.add_argument('--skip-cache-digests',action='store_true');args=p.parse_args()
 collection_gate();torch.set_num_threads(2);split=json.loads((RUN/'split_manifest.json').read_text());records=split['source_records'];assert_pins(records)
 if json.loads((DATA/'completed.json').read_text())['manifest_sha256']!=sha(RUN/'split_manifest.json'):raise ValueError('Cache manifest mismatch')
 raw=CupDataset(list(range(500)),raw_images=True);cache=CupDataset(list(range(500)));errors=[];pairs=0;started=time.monotonic()
 with np.load(RUN/'sampling_arrays.npz') as z:arrays={k:z[k] for k in z.files}
 for r in records:
  ep=r['episode_index'];mask=arrays['episode']==ep;approved=arrays['eligible_train'][mask]|arrays['eligible_validation'][mask]
  np.testing.assert_array_equal(approved,valid_anchors(arrays['expert'][mask],horizon=30))
  for cam in CAMERAS:
   path=DATA/cam/f'episode_{ep:06d}.npy';a=np.load(path,mmap_mode='r');audit=json.loads(path.with_suffix('.json').read_text())
   assert a.shape==(r['frames'],3,224,224) and a.dtype==np.float32
   if not args.skip_cache_digests:assert sha(path)==audit['cache_sha256']
   for i in [0,len(a)//2,len(a)-1]:assert np.isfinite(a[i]).all() and a[i].min()>=0 and a[i].max()<=1
   del a
  # Every episode: beginning, end, and the nearest approved frame to each intent transition.
  valid=np.flatnonzero(approved);frames={int(valid[0]),int(valid[-1])}
  for t in r['transitions']:frames.add(int(valid[np.argmin(abs(valid-t))]))
  for frame in sorted(frames):
   absolute=r['dataset_from_index']+frame;x=raw[absolute];y=cache[absolute]
   for name in ['action','observation.state','action_is_pad']:torch.testing.assert_close(x[name],y[name],rtol=0,atol=0)
   assert not y['action_is_pad'].any()
   for cam in CAMERAS:
    expected=F.resize(F.crop(x[cam].float()/255,*CROPS[cam]),[224,224]).clamp(0,1)
    errors.append(float((expected-y[cam]).abs().max()));torch.testing.assert_close(expected,y[cam],rtol=0,atol=0)
   pairs+=1
  if (ep+1)%10==0:
   status=dict(episodes=ep+1,target=500,paired_windows=pairs,elapsed_s=time.monotonic()-started);write('cache_verify_progress.json',status);print(json.dumps(status),flush=True)
 assert not np.isin(arrays['episode'][arrays['weights']>0],split['validation_episodes']).any()
 for pool,target in enumerate([.75,.25]):assert abs(arrays['weights'][arrays['source_pool']==pool].sum()-target)<1e-12
 write('cache_verified.json',dict(passed=True,cache_files=1500,episodes=500,paired_windows=pairs,max_pixel_error=max(errors),raw_action_state_exact=True,train_val_disjoint=True,complete_expert_chunks=True,digests_verified=not args.skip_cache_digests,manifest_sha256=sha(RUN/'split_manifest.json')))
 print((RUN/'cache_verified.json').read_text(),flush=True)
if __name__=='__main__':main()
