"""Reuse recent full image-cache verification only while all shared files remain unchanged."""
import json,os
import numpy as np
from pathlib import Path
from prepare import RUN,DATA,sha,write,collection_gate,assert_pins,CAMERAS
from rules import valid_anchors
OLD=Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_chunk30_20261028_002945')
def main():
 collection_gate()
 names=['split_manifest.json','train_stats.json','sampling_arrays.npz']
 for name in names:assert sha(RUN/name)==sha(OLD/name),name
 assert DATA.resolve()==(OLD/'cache').resolve()
 evidence=json.loads((OLD/'cache_verified.json').read_text())
 assert evidence['passed'] and evidence['digests_verified'] and evidence['manifest_sha256']==sha(RUN/'split_manifest.json')
 verified_at=(OLD/'cache_verified.json').stat().st_mtime_ns
 split=json.loads((RUN/'split_manifest.json').read_text());assert_pins(split['source_records'])
 cache_paths=[DATA/'lowdim.npz',DATA/'completed.json']
 for record in split['source_records']:
  for cam in CAMERAS:
   p=DATA/cam/f"episode_{record['episode_index']:06d}.npy"
   a=np.load(p,mmap_mode='r');assert a.shape==(record['frames'],3,224,224) and a.dtype==np.float32
   cache_paths.extend([p,p.with_suffix('.json')])
 for p in cache_paths:
  stat=p.stat();assert max(stat.st_mtime_ns,stat.st_ctime_ns)<=verified_at, f'Cache changed since full verification: {p}'
 with np.load(RUN/'sampling_arrays.npz') as a:
  assert not np.any(a['eligible_train']&a['eligible_validation'])
  for record in split['source_records']:
   start,end=record['dataset_from_index'],record['dataset_to_index']
   valid=valid_anchors(a['expert'][start:end],30)
   np.testing.assert_array_equal(valid,(a['eligible_train']|a['eligible_validation'])[start:end])
 evidence.update(inherited_digest_verification_from=str(OLD/'cache_verified.json'),unchanged_shared_cache_metadata_verified=True,unchanged_cache_files=len(cache_paths),chunk_size=30)
 write('cache_verified.json',evidence);print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
