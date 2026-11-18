"""Prepare only completed20Hz raw513, with fresh50-step masks and train-only stats."""
import datetime,hashlib,importlib.util,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from config import RUN,WORK
from rules import valid_anchors,source_balanced_weights,recovery_mask,validate_split,split_for_grid
SOURCE=Path('/home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031')
MANIFEST=SOURCE/'dataset_manifest.json'
HORIZON=50
FPS=20
CAMERAS=['observation.images.'+x for x in ['wrist','global','global_right']]
CROPS={CAMERAS[0]:(-140,0,640,640),CAMERAS[1]:(-80,0,640,640),CAMERAS[2]:(-80,0,640,640)}
def write(name,d):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def complete_windows(expert):return valid_anchors(np.asarray(expert,bool),HORIZON)
def quantile_stats(values,mask):
 x=np.asarray(values)[mask].astype(np.float64)
 if not len(x) or not np.isfinite(x).all():raise ValueError('No finite training expert values')
 return dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),q01=np.quantile(x,.01,axis=0).tolist(),q99=np.quantile(x,.99,axis=0).tolist(),count=[len(x)])
def assert_pins(records):
 for r in records:
  for name,h in r['source_hashes'].items():
   if sha(Path(r['source_path'])/name)!=h:raise ValueError('Raw source changed: '+str(Path(r['source_path'])/name))
def main():
 acceptance=json.loads((SOURCE/'completed.json').read_text());manifest=json.loads(MANIFEST.read_text())
 assert acceptance['completed'] and manifest['completed'] and manifest['total']==513 and manifest['fps']==20
 assert Counter(x['source'] for x in manifest['episodes'])=={'basic':400,'correction':100,'recovery':13}
 # Reuse pure gripper-intent labeling and independent20Hz recording-time audits.
 label_rules=module('basic_original_label_rules',WORK/'act_cup_renderfix500_chunk30_batch64_20261028_003937/rules.py')
 timing=module('record20_timing',SOURCE/'basic/audit.py')
 records=[];parts={k:[] for k in ['actions','states','episode','frame','expert','valid','key','recovery','source_pool','image_sizes','image_mtime_ns','image_ctime_ns','image_hashes']};start=0
 for index,item in enumerate(manifest['episodes']):
  root=Path(item['source_path']);summary=json.loads((root/'episode_summary.json').read_text());rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()];n=len(rows);timing.audit_rows(rows)
  assert summary['fps']==20 and summary['frame_count']==n and summary['dropped_frames']==0
  assert all(summary.get(k) is True for k in ['physical_grasp_ok','task_success','release_sequence_complete','placement_stable'])
  source=item['source'];grid=item['grid'];split=split_for_grid(grid);assert split==item['split'];meta={};expert=np.ones(n,bool);recover=np.zeros(n,bool);intervals=[]
  required=['frames.jsonl','episode_start.json','episode_summary.json']
  if source=='basic':labels,close,release=label_rules.full_labels(rows)
  else:
   required+=['correction.json','gripper_labels.json','training_mask.json','physics_status.jsonl']
   meta=json.loads((root/'correction.json').read_text());assert meta['accepted'] and meta['recovery_complete']
   labels=np.asarray(json.loads((root/'gripper_labels.json').read_text())['labels'],np.float32)
   expert=np.asarray(json.loads((root/'training_mask.json').read_text())['expert_frame_mask'],bool)
   assert expert.shape==(n,) and labels.shape==(n,) and np.isin(labels,[0,1]).all()
   recover,intervals=recovery_mask(rows,meta,expert)
  transitions=(np.flatnonzero(np.diff(labels)!=0)+1).tolist();key=np.zeros(n,bool)
  for t in transitions:key[max(0,t-16):min(n,t+16)]=True
  states=np.asarray([x['observation.state'] for x in rows],np.float32);actions=np.asarray([x['action'] for x in rows],np.float32)
  assert states.shape==actions.shape==(n,7) and np.isfinite(states).all() and np.isfinite(actions).all();actions[:,-1]=labels
  valid=complete_windows(expert)
  sizes=np.zeros((n,3),np.int64);mtimes=sizes.copy();ctimes=sizes.copy();hashes=np.zeros((n,3),dtype='S64')
  for i,row in enumerate(rows):
   for j,cam in enumerate(CAMERAS):
    image=root/row[cam];assert image.resolve().is_relative_to(root.resolve());stat=image.stat();assert stat.st_size>100
    sizes[i,j]=stat.st_size;mtimes[i,j]=stat.st_mtime_ns;ctimes[i,j]=stat.st_ctime_ns;hashes[i,j]=sha(image).encode()
  source_name='actual_policy_push_recovery' if source=='recovery' else source
  r=dict(episode_index=index,source=source_name,source_path=str(root),frames=n,grid=grid,seed=item['seed'],split=split,task=rows[0]['task'],task_id=meta.get('task_id'),kind=meta.get('task',{}).get('kind'),transitions=transitions,dataset_from_index=start,dataset_to_index=start+n,source_hashes={name:sha(root/name) for name in required},expert_frames=int(expert.sum()),eligible_windows=int(valid.sum()),recovery_frames=int(recover.sum()),recovery_intervals=intervals,reference_basic_episode=meta.get('task',{}).get('reference_basic_episode'))
  r['source_frames_sha256']=r['source_hashes']['frames.jsonl'];records.append(r)
  values=dict(actions=actions,states=states,episode=np.full(n,index,np.int32),frame=np.arange(n,dtype=np.int32),expert=expert,valid=valid,key=key,recovery=recover,source_pool=np.full(n,source!='basic',np.uint8),image_sizes=sizes,image_mtime_ns=mtimes,image_ctime_ns=ctimes,image_hashes=hashes)
  for k,v in values.items():parts[k].append(v)
  start+=n
  if index%10==0:print('PREPARED',index+1,'/513',start,'frames',flush=True)
 arrays={k:np.concatenate(v) for k,v in parts.items()};validate_split(records)
 train=[r['episode_index'] for r in records if r['split']=='train'];val=[r['episode_index'] for r in records if r['split']=='validation'];assert len(train)==461 and len(val)==52
 training=np.isin(arrays['episode'],train);eligible=training&arrays['valid'];heldout=~training&arrays['valid'];w=source_balanced_weights(eligible,arrays['key'],arrays['source_pool'],arrays['recovery'])
 arrays.update(eligible_train=eligible,eligible_validation=heldout,weights=w)
 stats={name:quantile_stats(arrays[k],training&arrays['expert']) for name,k in [('action','actions'),('observation.state','states')]}
 assert stats['action']['q01'][-1]==0 and stats['action']['q99'][-1]==1
 for c in CAMERAS:stats[c]=dict(mean=[[[.5]]]*3,std=[[[.5]]]*3,min=[[[0.]]]*3,max=[[[1.]]]*3)
 np.savez(RUN/'lowdim.npz',**{k:arrays[k] for k in ['actions','states','episode','frame']})
 np.savez_compressed(RUN/'sampling_arrays.npz',**{k:arrays[k] for k in ['episode','frame','expert','key','recovery','source_pool','eligible_train','eligible_validation','weights']})
 np.savez(RUN/'image_pins.npz',**{k:arrays[k] for k in ['image_sizes','image_mtime_ns','image_ctime_ns','image_hashes']})
 write('split_manifest.json',dict(train_episodes=train,validation_episodes=val,source_records=records,holdout_cells=[[1,1],[2,3]],recording_manifest_sha256=sha(MANIFEST)))
 write('train_stats.json',stats)
 audit=dict(episodes=513,frames=start,fps=20,horizon=50,train_episodes=len(train),validation_episodes=len(val),source_counts=dict(Counter(r['source'] for r in records)),training_windows=int(eligible.sum()),validation_windows=int(heldout.sum()),source_mass={'basic':float(w[arrays['source_pool']==0].sum()),'correction':float(w[arrays['source_pool']==1].sum())},recovery_multiplier=3.,transition_multiplier=1.5,transition_radius_frames=16,transition_radius_s=.8,excluded_nonexpert_frames=int((~arrays['expert']).sum()),normalization_rule='train episodes AND expert frames only',all_source_jpegs_sha256_pinned=True)
 write('sampling_audit.json',audit)
 files=[RUN/n for n in ['split_manifest.json','lowdim.npz','sampling_arrays.npz','image_pins.npz','train_stats.json','sampling_audit.json']]+[MANIFEST,SOURCE/'completed.json']
 write('preparation_complete.json',dict(passed=True,episodes=513,accepted_new=513,fps=20,chunk_size=50,train_episodes=461,validation_episodes=52,files={str(p):sha(p) for p in files}))
 print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
