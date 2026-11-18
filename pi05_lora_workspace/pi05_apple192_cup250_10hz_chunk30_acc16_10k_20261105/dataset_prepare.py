"""Immutable 10Hz two-object subset, task-balanced sampling, raw-image pins."""
import json,hashlib,copy,datetime
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from config import RUN,WORK
from rules import valid_anchors
FPS=10
HORIZON=30
CAMERAS=['observation.images.'+x for x in ['wrist','global','global_right']]
CROPS={CAMERAS[0]:(-140,0,640,640),CAMERAS[1]:(-80,0,640,640),CAMERAS[2]:(-80,0,640,640)}
def write(name,data):
 p=RUN/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def assert_pins(records):
 for r in records:
  for name,digest in r['source_hashes'].items():
   if sha(Path(r['source_path'])/name)!=digest:raise ValueError('Source changed: '+str(Path(r['source_path'])/name))
def quantile_stats(values,mask):
 x=values[mask].astype(np.float64);assert len(x) and np.isfinite(x).all()
 return dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),q01=np.quantile(x,.01,axis=0).tolist(),q99=np.quantile(x,.99,axis=0).tolist(),count=[len(x)])
def select_cup(records):
 basic=[]
 for grid in sorted({tuple(x['grid']) for x in records[:400]}):
  group=[x for x in records[:400] if tuple(x['grid'])==grid];assert len(group)==20
  basic += [group[i]['episode_index'] for i in range(0,20,2)]
 corr=records[400:500]
 from scipy.optimize import milp,Bounds,LinearConstraint
 matrix=[];lower=[];upper=[]
 def add(v,lo,hi):matrix.append(v);lower.append(lo);upper.append(hi)
 add(np.ones(100),50,50)
 for grid in sorted({tuple(x['grid']) for x in corr}):add([tuple(x['grid'])==grid for x in corr],2,3)
 for kind in sorted({x['kind'] for x in corr}):
  mask=np.array([x['kind']==kind for x in corr]);n=mask.sum();add(mask,n//2,(n+1)//2)
 add([x['split']=='validation' for x in corr],5,5)
 result=milp(np.random.default_rng(20261105).uniform(size=100),integrality=np.ones(100),bounds=Bounds(0,1),constraints=LinearConstraint(np.array(matrix,float),lower,upper))
 assert result.success,result.message
 ids=basic+[x['episode_index'] for x,v in zip(corr,result.x) if v>.5];assert len(ids)==250
 return ids

def main():
 apple=WORK/'act_apple_cached_20261018';cup=WORK/'pi05_cup_recovery513_lora_20261029'
 ar=json.loads((apple/'split_manifest.json').read_text())['source_records'];cr=json.loads((cup/'split_manifest.json').read_text())['source_records'];assert len(ar)==192
 appledata=np.load('/home/lp531/smolvla_data/act_apple_rgb224_cache_20261018/lowdim.npz');cupdata=np.load(cup/'lowdim.npz');cupmask=np.load(cup/'sampling_arrays.npz')
 chosen=[('apple',x,appledata) for x in ar]+[('cup',cr[i],cupdata) for i in select_cup(cr)]
 records=[];parts=defaultdict(list);start=0
 for obj,old,data in chosen:
  root=Path(old['source_path']);rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()];n=len(rows);summary=json.loads((root/'episode_summary.json').read_text());assert n==old['frames'] and summary['fps']==10 and summary['dropped_frames']==0
  assert sha(root/'frames.jsonl')==old['source_frames_sha256']
  a,b=old['dataset_from_index'],old['dataset_to_index'];actions=data['actions'][a:b].copy();states=data['states'][a:b].copy();assert actions.shape==states.shape==(n,7)
  assert np.isin(actions[:,-1],[0,1]).all() and np.isfinite(actions).all() and np.isfinite(states).all()
  assert np.array_equal(states,np.array([x['observation.state'] for x in rows],np.float32))
  assert np.array_equal(actions[:,:6],np.array([x['action'][:6] for x in rows],np.float32))
  expert=np.ones(n,bool) if obj=='apple' else cupmask['expert'][a:b].copy()
  source=('apple_full' if old['source']=='original' else 'apple_correction') if obj=='apple' else 'cup_'+old['source'];assert source in ['apple_full','apple_correction','cup_basic','cup_correction']
  task='Physically grasp the apple and lift it clear of the table.' if source=='apple_correction' else old['task']
  required=['frames.jsonl','episode_start.json','episode_summary.json']
  if source.endswith('correction'):required+=['correction.json','gripper_labels.json']
  if obj=='cup' and source.endswith('correction'):required+=['training_mask.json','physics_status.jsonl']
  pins={k:sha(root/k) for k in required}
  for k,h in old.get('source_hashes',{}).items():
   assert sha(root/k)==h
  idx=len(records);r=dict(episode_index=idx,source=source,object=obj,source_path=str(root),frames=n,grid=old['grid'],seed=old['seed'],split=old['split'],task=task,original_task=old['task'],original_episode_index=old['episode_index'],dataset_from_index=start,dataset_to_index=start+n,source_hashes=pins,kind=old.get('kind'))
  records.append(r)
  transitions=np.flatnonzero(np.diff(actions[:,-1])!=0)+1;key=np.zeros(n,bool)
  for t in transitions:key[max(0,t-8):min(n,t+8)]=True
  sizes=np.empty((n,3),np.int64);mtime=sizes.copy();ctime=sizes.copy();hashes=np.empty((n,3),dtype='S64')
  for i,row in enumerate(rows):
   assert row['target_object']==obj
   for j,cam in enumerate(CAMERAS):
    path=root/row[cam];assert path.resolve().is_relative_to(root.resolve());st=path.stat();sizes[i,j]=st.st_size;mtime[i,j]=st.st_mtime_ns;ctime[i,j]=st.st_ctime_ns;hashes[i,j]=sha(path).encode()
  for k,v in dict(actions=actions,states=states,episode=np.full(n,idx),frame=np.arange(n),expert=expert,key=key,valid=valid_anchors(expert,HORIZON),source_pool=np.full(n,['apple_full','apple_correction','cup_basic','cup_correction'].index(source)),image_sizes=sizes,image_mtime_ns=mtime,image_ctime_ns=ctime,image_hashes=hashes).items():parts[k].append(v)
  start+=n
  if idx%25==0:print('PINNED',idx+1,source,start,flush=True)
 arrays={k:np.concatenate(v) for k,v in parts.items()};train=[x['episode_index'] for x in records if x['split']=='train'];val=[x['episode_index'] for x in records if x['split']=='validation'];assert len(train)==395 and len(val)==47
 assert not {x['seed'] for x in records if x['split']=='train'} & {x['seed'] for x in records if x['split']=='validation'}
 training=np.isin(arrays['episode'],train);eligible=training&arrays['valid'];validation=~training&arrays['valid'];weights=np.zeros(start)
 for pool,mass in enumerate([.35,.15,.375,.125]):
  mask=eligible&(arrays['source_pool']==pool);assert mask.any();m=np.where(arrays['key'][mask],1.5,1.);weights[mask]=mass*m/m.sum()
 assert np.isclose(weights.sum(),1)
 stats={name:quantile_stats(arrays[key],training&arrays['expert']) for name,key in [('action','actions'),('observation.state','states')]}
 assert stats['action']['q01'][-1]==0 and stats['action']['q99'][-1]==1
 for cam in CAMERAS:stats[cam]=dict(mean=[[[.5]]]*3,std=[[[.5]]]*3,min=[[[0.]]]*3,max=[[[1.]]]*3)
 np.savez(RUN/'lowdim.npz',**{k:arrays[k] for k in ['actions','states','episode','frame']})
 np.savez_compressed(RUN/'image_pins.npz',**{k:arrays[k] for k in ['image_sizes','image_mtime_ns','image_ctime_ns','image_hashes']})
 np.savez_compressed(RUN/'sampling_arrays.npz',episode=arrays['episode'],frame=arrays['frame'],expert=arrays['expert'],key=arrays['key'],source_pool=arrays['source_pool'],eligible_train=eligible,eligible_validation=validation,weights=weights)
 write('split_manifest.json',dict(train_episodes=train,validation_episodes=val,source_records=records,holdout_cells=[[1,1],[2,3]],split_rule='Preserve each source dataset spatial holdout; apple/cup physical grid coverage differs'))
 write('train_stats.json',stats)
 write('sampling_audit.json',dict(episodes=len(records),frames=start,sources=dict(Counter(x['source'] for x in records)),pool_probability=[float(weights[arrays['source_pool']==i].sum()) for i in range(4)],training_windows=int(eligible.sum()),validation_windows=int(validation.sum()),apple_partial_task_relabelled=59,nonexpert_excluded=int((~arrays['expert']).sum()),transition_radius_frames=8))
 files=['lowdim.npz','image_pins.npz','sampling_arrays.npz','split_manifest.json','train_stats.json','sampling_audit.json']
 write('preparation_complete.json',dict(passed=True,fps=10,chunk_size=30,episodes=442,accepted_new=442,files={str(RUN/f):sha(RUN/f) for f in files}))
 print((RUN/'sampling_audit.json').read_text(),flush=True)
if __name__=='__main__':main()
