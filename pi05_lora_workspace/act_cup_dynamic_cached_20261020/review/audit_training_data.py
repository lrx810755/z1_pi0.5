import json, hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
run=Path('/home/lp531/pi05_lora_workspace/act_cup_dynamic_cached_20261020');rec=json.loads((run/'split_manifest.json').read_text())['source_records'];sampling=np.load(run/'sampling_arrays.npz');low=np.load('/home/lp531/smolvla_data/act_cup_dynamic_rgb224_cache_20261020/lowdim.npz')
summary={};issues=[];mass=defaultdict(float);dur=defaultdict(list);age=defaultdict(list);sync=defaultdict(list);steps=defaultdict(list);seed=defaultdict(set);tail=[];phase=defaultdict(float);lowdim_mismatch=0
for r in rec:
 p=Path(r['source_path']);raw=(p/'frames.jsonl').read_bytes();rows=[json.loads(l) for l in raw.splitlines()];source=r['source'];n=len(rows);c=r['close_frame'];o=r['open_frame'];start=r['dataset_from_index'];w=sampling['weights'][start:start+n];a=np.array([f['action'] for f in rows]);s=np.array([f['observation.state'] for f in rows]);label=np.array([int(i>=c and (o is None or i<o)) for i in range(n)]);aa=a.copy();aa[:,-1]=label
 if hashlib.sha256(raw).hexdigest()!=r['source_frames_sha256']:issues.append(['changed_source',str(p)])
 if not np.array_equal(aa.astype(low['actions'].dtype),low['actions'][start:start+n]) or not np.array_equal(s.astype(low['states'].dtype),low['states'][start:start+n]):lowdim_mismatch+=1
 seed[r['split']].add(r['seed']);dur[source].append(n/10)
 for i,f in enumerate(rows):
  age[source].append(f['action_age_s']);stamps=[f[k] for k in ['wrist_image_stamp','global_image_stamp','global_right_image_stamp']];sync[source].append(max(stamps)-min(stamps));phase[f['phase']]+=float(w[i]);stage='before_close' if i<c else ('held' if o is None or i<o else 'release_and_after');mass[stage]+=float(w[i])
  if c-15<=i<c:mass['chunk_contains_close_transition']+=float(w[i])
  if o is not None and o-15<=i<o:mass['chunk_contains_release_transition']+=float(w[i])
  if o is not None and o-20<=i<=o+10:mass['within_2s_before_1s_after_release']+=float(w[i])
  if w[i]>0:
   for j in range(16):mass['target_closed' if label[i+j] else 'target_open']+=float(w[i])/16
 if r['split']=='train' and o is not None:tail.append(n-o)
 steps[source].extend(np.max(abs(np.diff(a[:,:6],axis=0)),axis=1).tolist())
summary['source_episode_counts']=dict(Counter(r['source'] for r in rec));summary['source_hash_changes']=issues;summary['lowdim_mismatch_episodes']=lowdim_mismatch;summary['train_validation_seed_overlap']=sorted(seed['train']&seed['validation']);summary['weighted_sampling_mass']=dict(mass);summary['weighted_phase_mass']=dict(phase);summary['train_original_release_tail_frames']={'min':min(tail),'median':float(np.median(tail)),'max':max(tail),'less_than_16':sum(x<16 for x in tail)}
for name,values in [('duration_seconds',dur),('action_age_seconds',age),('camera_interstamp_gap_seconds',sync),('max_arm_command_step_rad',steps)]:summary[name]={k:{'min':float(np.min(v)),'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'max':float(np.max(v))} for k,v in values.items()}
summary['correction_open_frames']=sum(r['open_frame'] is not None for r in rec if r['source']!='original')
out=run/'review/training_data_audit.json';out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
