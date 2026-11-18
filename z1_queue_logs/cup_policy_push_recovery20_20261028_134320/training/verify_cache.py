"""Verify all new digests, inherited immutable crops, horizon30 and paired pixels."""
import numpy as np
from prepare import RUN,DATA,NEW,CAMERAS,CROPS,load,sha,write,collection_gate,assert_pins
from build_cache import inherited_cache

def main():
 import torch
 from torchvision.transforms import functional as F
 from cache_data import CupDataset
 from rules import valid_anchors
 collection_gate();torch.set_num_threads(2);split=load(RUN/'split_manifest.json');records=split['source_records'];assert_pins(records)
 if split['completion_evidence_sha256']!=sha(NEW/'completion_verified.json'):raise ValueError('Acceptance changed after prep')
 if load(DATA/'completed.json')['manifest_sha256']!=sha(RUN/'split_manifest.json'):raise ValueError('Cache manifest differs')
 inherited=inherited_cache(records)
 raw=CupDataset(list(range(len(records))),raw_images=True);cached=CupDataset(list(range(len(records))))
 with np.load(RUN/'sampling_arrays.npz') as a:arrays={k:a[k] for k in a.files}
 pairs=0;max_error=0.
 for r in records:
  ep=r['episode_index'];start,end=r['dataset_from_index'],r['dataset_to_index'];valid=arrays['eligible_train'][start:end]|arrays['eligible_validation'][start:end]
  np.testing.assert_array_equal(valid,valid_anchors(arrays['expert'][start:end],30))
  if not valid.any():raise ValueError('Episode has no approved expert30 window')
  if ep<500:continue # prior full digest + paired pixel verification inherited only while unchanged
  for cam in CAMERAS:
   path=DATA/cam/f'episode_{ep:06d}.npy';audit=load(path.with_suffix('.json'));pixels=np.load(path,mmap_mode='r')
   if pixels.shape!=(r['frames'],3,224,224) or pixels.dtype!=np.float32 or sha(path)!=audit['cache_sha256']:raise ValueError('New cache shape/digest failed')
   for image in audit['source_images']:
    if sha(__import__('pathlib').Path(r['source_path'])/image['path'])!=image['sha256']:raise ValueError('New source image changed')
  indices=np.flatnonzero(valid);frames={int(indices[0]),int(indices[-1])}
  for t in r['transitions']:frames.add(int(indices[np.argmin(abs(indices-t))]))
  for f in sorted(frames):
   absolute=start+f;x=raw[absolute];y=cached[absolute]
   for name in ['action','observation.state','action_is_pad']:torch.testing.assert_close(x[name],y[name],rtol=0,atol=0)
   if y['action_is_pad'].any():raise ValueError('Unexpected padding')
   for cam in CAMERAS:
    expected=F.resize(F.crop(x[cam].float()/255,*CROPS[cam]),[224,224]).clamp(0,1)
    max_error=max(max_error,float((expected-y[cam]).abs().max()));torch.testing.assert_close(expected,y[cam],rtol=0,atol=0)
   pairs+=1
 if np.any(arrays['weights'][arrays['eligible_validation']]!=0):raise ValueError('Heldout sampling leak')
 for pool,mass in enumerate([.75,.25]):np.testing.assert_allclose(arrays['weights'][arrays['source_pool']==pool].sum(),mass,atol=1e-12)
 write('cache_verified.json',dict(passed=True,digests_verified=True,inherited500=inherited,new_paired_windows=pairs,max_pixel_error=max_error,complete_expert_chunks=True,train_val_disjoint=True,manifest_sha256=sha(RUN/'split_manifest.json')))
if __name__=='__main__':main()
