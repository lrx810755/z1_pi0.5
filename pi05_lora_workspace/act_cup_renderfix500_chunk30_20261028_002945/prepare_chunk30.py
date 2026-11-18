"""Reuse exact image cache and spatial split; rebuild only complete30 expert anchors."""
import json
import numpy as np
from pathlib import Path
from prepare import RUN, DATA, write, sha, collection_gate, assert_pins
from rules import valid_anchors
from weights import source_balanced_weights
BASE=Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_cached_20261027')
def main():
 collection_gate()
 assert sha(RUN/'split_manifest.json')==sha(BASE/'split_manifest.json')
 assert sha(RUN/'train_stats.json')==sha(BASE/'train_stats.json')
 assert DATA.resolve()==(BASE/'cache').resolve()
 split=json.loads((RUN/'split_manifest.json').read_text());assert_pins(split['source_records'])
 old=json.loads((BASE/'cache_verified.json').read_text())
 assert old['passed'] and old['digests_verified'] and old['manifest_sha256']==sha(RUN/'split_manifest.json')
 with np.load(BASE/'sampling_arrays.npz') as a:arrays={k:a[k] for k in a.files}
 valid=np.zeros(len(arrays['episode']),bool);counts=[]
 for record in split['source_records']:
  start,end=record['dataset_from_index'],record['dataset_to_index']
  assert np.all(arrays['episode'][start:end]==record['episode_index'])
  anchors=valid_anchors(arrays['expert'][start:end],horizon=30)
  valid[start:end]=anchors
  assert anchors.any(),record['source_path']
  # Exhaustive safety check: no anchor crosses an episode boundary or an intervention.
  for local in np.flatnonzero(anchors):
   assert local+30<=end-start and arrays['expert'][start+local:start+local+30].all()
  counts.append(dict(episode=record['episode_index'],source=record['source'],split=record['split'],complete30_windows=int(anchors.sum())))
 train=np.isin(arrays['episode'],split['train_episodes'])
 arrays['eligible_train']=train&valid;arrays['eligible_validation']=~train&valid
 arrays['weights']=source_balanced_weights(arrays['eligible_train'],arrays['key'],arrays['source_pool'])
 assert not np.any(arrays['eligible_train']&arrays['eligible_validation'])
 assert not np.any(arrays['weights'][~arrays['eligible_train']])
 np.savez_compressed(RUN/'sampling_arrays.npz',**arrays)
 audit=json.loads((BASE/'sampling_audit.json').read_text())
 audit.update(chunk_size=30,training_windows=int(arrays['eligible_train'].sum()),validation_windows=int(arrays['eligible_validation'].sum()),shared_image_cache=str(DATA.resolve()),previous_chunk_size=16)
 write('sampling_audit.json',audit);write('horizon30_windows_verified.json',dict(passed=True,chunk_size=30,all_windows_expert=True,no_cross_episode=True,per_episode=counts,training_windows=audit['training_windows'],validation_windows=audit['validation_windows']))
 write('cache_reuse_provenance.json',dict(base_run=str(BASE),shared_cache=str(DATA.resolve()),manifest_sha256=sha(RUN/'split_manifest.json'),original_cache_verification=old,full_cache_digest_reverification_required=True))
 print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
