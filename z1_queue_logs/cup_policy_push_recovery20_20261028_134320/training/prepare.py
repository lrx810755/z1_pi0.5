"""Freeze verified existing500 plus independently accepted actual policy recoveries."""
import argparse,copy,hashlib,json,os,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from rules import *
RUN=Path(__file__).resolve().parent
DATA=RUN/'cache'
OLD=Path('/home/lp531/pi05_lora_workspace/act_cup_renderfix500_chunk30_batch64_20261028_003937')
NEW=Path(os.environ.get('Z1_PUSH_RECOVERY_DATA','/home/lp531/smolvla_data/z1_cup_policy_push_recovery20_20261028'))
CAMERAS=['observation.images.'+x for x in ['wrist','global','global_right']]
CROPS={CAMERAS[0]:(-140,0,640,640),CAMERAS[1]:(-80,0,640,640),CAMERAS[2]:(-80,0,640,640)}
REQUIRED=['frames.jsonl','episode_summary.json','episode_start.json','correction.json','gripper_labels.json','training_mask.json','physics_status.jsonl']

def write(name,value):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str));t.replace(p)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def load(path):return json.loads(Path(path).read_text())

def assert_pins(records):
 for r in records:
  for name,digest in r['source_hashes'].items():
   if sha(Path(r['source_path'])/name)!=digest:raise ValueError(f'Source changed: {r["source_path"]}/{name}')

def assert_prepared():
 complete=load(RUN/'preparation_complete.json');manifest=load(RUN/'split_manifest.json')
 if complete['manifest_sha256']!=sha(RUN/'split_manifest.json'):raise ValueError('Frozen preparation manifest changed')
 if manifest['completion_evidence_sha256']!=sha(NEW/'completion_verified.json'):raise ValueError('Acceptance changed after preparation')
 assert_pins(manifest['source_records'])


def completion_gate(root=NEW):
 root=Path(root).resolve();evidence=load(root/'completion_verified.json')
 if evidence.get('passed') is not True or evidence.get('source_type')!='actual_policy_push_recovery':raise ValueError('Independent completed acceptance evidence required')
 count=evidence.get('accepted_count');paths=[Path(p).resolve() for p in evidence.get('episodes',[])]
 if type(count) is not int or not 16<=count<=20 or len(paths)!=count or len(set(paths))!=count:raise ValueError('Final verified count must match16..20 distinct accepted episode paths')
 if any(not p.is_relative_to(root) or not p.is_dir() for p in paths):raise ValueError('Verified episode escapes dataset root')
 # Reject undeclared episode directories, including accidentally admitted failed attempts.
 if set(root.glob('episode_[0-9]*'))!=set(paths):raise ValueError('Dataset episode directories differ from verified accepted list')
 metas=[load(p/'correction.json') for p in paths];validate_new_collection(metas,count)
 return paths,evidence

def collection_gate():
 records=load(OLD/'split_manifest.json')['source_records'];validate_old_records(records);assert_pins(records)
 paths,evidence=completion_gate()
 return records,paths,evidence

def old_arrays_and_annotations(records):
 with np.load(OLD/'cache'/'lowdim.npz') as a:parts={k:[a[k].copy()] for k in ['actions','states','episode','frame']}
 with np.load(OLD/'sampling_arrays.npz') as a:expert=a['expert'].copy()
 parts.update(expert=[expert],key=[],recovery=[],source_pool=[],valid=[])
 for r in records:
  start,end=r['dataset_from_index'],r['dataset_to_index'];n=r['frames'];root=Path(r['source_path'])
  rows=[json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
  key=np.zeros(n,bool)
  for t in r['transitions']:key[max(0,t-8):min(n,t+8)]=True
  localexpert=expert[start:end];recover=np.zeros(n,bool);intervals=[]
  if r['source']=='correction':
   meta=load(root/'correction.json')
   if meta.get('accepted') is not True or meta.get('recovery_complete') is not True:raise ValueError('Old correction acceptance changed')
   recover,intervals=recovery_mask(rows,meta,localexpert)
  r['recovery_intervals']=intervals;r['recovery_frames']=int(recover.sum())
  r['eligible_windows']=int(valid_anchors(localexpert).sum())
  for k,v in dict(key=key,recovery=recover,source_pool=np.full(n,int(r['source']=='correction'),np.uint8),valid=valid_anchors(localexpert)).items():parts[k].append(v)
 return parts

def new_record(path,idx,start):
 hashes={name:sha(path/name) for name in REQUIRED};summary=load(path/'episode_summary.json');meta=load(path/'correction.json')
 for k in ['physical_grasp_ok','task_success','release_sequence_complete','placement_stable']:
  if summary.get(k) is not True:raise ValueError(f'{path}: unsuccessful {k}')
 if summary.get('target_object')!='cup' or summary.get('dropped_frames')!=0:raise ValueError('Wrong target/dropped frames')
 rows=[json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()];n=len(rows)
 if n!=summary['frame_count'] or n<30:raise ValueError('Invalid frame count')
 labels=np.asarray(load(path/'gripper_labels.json')['labels'],np.float32);expert=np.asarray(load(path/'training_mask.json')['expert_frame_mask'])
 if labels.shape!=(n,) or not np.isin(labels,[0,1]).all() or expert.shape!=(n,) or expert.dtype!=bool:raise ValueError('Aligned binary labels/expert mask required')
 valid=valid_anchors(expert)
 if not valid.any():raise ValueError('No complete30-action expert window')
 if meta.get('table_contact') or meta.get('dynamic_violations'):raise ValueError('Unsafe physical recovery')
 final=meta['final_status']
 for k in ['task_success','physical_grasp_ok','released','placement_stable','release_sequence_complete','target_collisions_enabled']:
  if final.get(k) is not True:raise ValueError('Final physics failure '+k)
 if final.get('frozen') is not False or final.get('holding_observed') is not False:raise ValueError('Invalid final physical state')
 physics=[json.loads(line) for line in (path/'physics_status.jsonl').read_text().splitlines()]
 if len(physics)!=meta.get('dynamic_samples') or not physics:raise ValueError('Incomplete physics evidence')
 previous=-np.inf
 for sample in physics:
  t=sample['simulation_time'];status=sample['status']
  if not np.isfinite(t) or t<previous:raise ValueError('Invalid physics times')
  previous=t
  for k,v in [('frozen',False),('target_collisions_enabled',True),('left_finger_touching_table',False),('right_finger_touching_table',False)]:
   if status.get(k) is not v:raise ValueError('Unsafe/missing physics '+k)
 grid=summary['scene_randomization']['target_grid'];seed=summary['scene_randomization']['seed'];split=split_for_grid(grid)
 source_grid=meta['source_grid']
 if meta.get('seed',seed)!=seed or meta.get('split',split)!=split or tuple(source_grid)!=tuple(grid):raise ValueError('Source layout/seed/split mismatch')
 action=np.asarray([r['action'] for r in rows],np.float32);state=np.asarray([r['observation.state'] for r in rows],np.float32)
 if action.shape!=(n,7) or state.shape!=(n,7) or not np.isfinite(action).all() or not np.isfinite(state).all():raise ValueError('Invalid lowdim values')
 action[:,-1]=labels;transitions=(np.flatnonzero(np.diff(labels)!=0)+1).tolist();key=np.zeros(n,bool)
 for t in transitions:key[max(0,t-8):min(n,t+8)]=True
 for i,row in enumerate(rows):
  if row['frame_index']!=i or row['target_object']!='cup':raise ValueError('Invalid frame alignment')
  for cam in CAMERAS:
   image=path/row[cam]
   if not image.resolve().is_relative_to(path) or not image.is_file() or not image.stat().st_size:raise ValueError('Missing/unsafe camera path')
 recovery,intervals=recovery_mask(rows,meta,expert)
 if not (recovery&valid).any():raise ValueError('No eligible recovery anchors')
 record=dict(episode_index=idx,source='actual_policy_push_recovery',source_type=meta['source_type'],source_path=str(path),frames=n,grid=grid,source_grid=source_grid,seed=seed,split=split,task=rows[0]['task'],source_identity=meta['source_identity'],task_id=meta.get('task_id'),kind='actual_policy_push_recovery',transitions=transitions,dataset_from_index=start,dataset_to_index=start+n,source_hashes=hashes,source_frames_sha256=hashes['frames.jsonl'],expert_frames=int(expert.sum()),eligible_windows=int(valid.sum()),recovery_frames=int(recovery.sum()),recovery_intervals=intervals)
 return record,dict(actions=action,states=state,episode=np.full(n,idx,np.int32),frame=np.arange(n,dtype=np.int32),expert=expert,valid=valid,key=key,recovery=recovery,source_pool=np.ones(n,np.uint8))

def assemble(records,parts):
 arrays={k:np.concatenate(v) for k,v in parts.items()};validate_split(records)
 train=[r['episode_index'] for r in records if r['split']=='train'];val=[r['episode_index'] for r in records if r['split']=='validation']
 training=np.isin(arrays['episode'],train);eligible=training&arrays['valid'];validation=~training&arrays['valid']
 w=source_balanced_weights(eligible,arrays['key'],arrays['source_pool'],arrays['recovery']);arrays.update(eligible_train=eligible,eligible_validation=validation,weights=w)
 phase={}
 for pool,name in enumerate(['basic','correction']):
  mask=eligible&(arrays['source_pool']==pool)
  phase[name]={}
  for phase_name,selection in [('recovery',mask&arrays['recovery']),('transition_only',mask&arrays['key']&~arrays['recovery']),('ordinary',mask&~arrays['key']&~arrays['recovery'])]:
   phase[name][phase_name]=dict(anchors=int(selection.sum()),global_mass=float(w[selection].sum()),within_pool_mass=float(w[selection].sum()/[.75,.25][pool]))
 audit=dict(episodes=len(records),frames=len(w),source_counts=dict(Counter(r['source'] for r in records)),train_episodes=len(train),validation_episodes=len(val),training_windows=int(eligible.sum()),validation_windows=int(validation.sum()),source_mass={name:float(w[arrays['source_pool']==p].sum()) for p,name in enumerate(['basic','correction'])},phase_sampling_mass=phase,recovery_multiplier=3.,transition_multiplier=1.5,ordinary_multiplier=1.,overlap_rule='maximum',horizon=30,excluded_nonexpert_frames=int((~arrays['expert']).sum()),normalization_rule='train episodes AND expert frames; exclude policy/intervention/heldout')
 return arrays,train,val,audit,training

def recovery_kind_audit(records,arrays):
 report={}
 for kind in sorted({r['kind'] for r in records if r['source']=='correction'}):
  selected=[r for r in records if r['kind']==kind];mask=np.isin(arrays['episode'],[r['episode_index'] for r in selected])&arrays['eligible_train'];recovery=mask&arrays['recovery']
  examples=[]
  for r in selected:
   start,end=r['dataset_from_index'],r['dataset_to_index'];local=arrays['recovery'][start:end];ids=np.flatnonzero(local)
   examples.append(dict(episode_index=r['episode_index'],source_path=r['source_path'],split=r['split'],expert_recovery_union_frames=int(local.sum()),expert_recovery_union_duration_s=round(float(local.sum()/10),3),recorded_and_extended_intervals=r['recovery_intervals']))
  report[kind]=dict(episodes=len(selected),train_eligible=int(mask.sum()),recovery_anchors=int(recovery.sum()),recovery_mass_global=float(arrays['weights'][recovery].sum()),within_correction_mass=float(arrays['weights'][recovery].sum()/.25),origins=sorted({i['origin'] for r in selected for i in r['recovery_intervals']}),episodes_detail=examples)
 return report


def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-old',action='store_true');p.add_argument('--audit-only',action='store_true');args=p.parse_args()
 if (RUN/'preparation_complete.json').exists() and not(args.audit_old or args.audit_only):
  collection_gate();assert_prepared();print('Frozen preparation verified');return
 records=copy.deepcopy(load(OLD/'split_manifest.json')['source_records']);validate_old_records(records);assert_pins(records);parts=old_arrays_and_annotations(records)
 if args.audit_old:
  arrays,_,_,audit,_=assemble(records,parts);write('old500_recovery_by_kind.json',recovery_kind_audit(records,arrays));write('old500_recovery_audit.json',audit);print(json.dumps(audit,indent=2));return
 _,paths,evidence=collection_gate();start=records[-1]['dataset_to_index']
 for path in paths:
  record,values=new_record(path,len(records),start);records.append(record)
  for k,v in values.items():parts[k].append(v)
  start+=record['frames']
 arrays,train,val,audit,training=assemble(records,parts);audit['completion_evidence_sha256']=sha(NEW/'completion_verified.json')
 if args.audit_only:write('source_audit.json',audit);print(json.dumps(audit,indent=2));return
 DATA.mkdir(exist_ok=True)
 for r in records:
  pin=RUN/'source_pins'/f'episode_{r["episode_index"]:06d}';pin.mkdir(parents=True,exist_ok=True)
  for name in r['source_hashes']:shutil.copyfile(Path(r['source_path'])/name,pin/name)
 np.savez(DATA/'lowdim.npz',**{k:arrays[k] for k in ['actions','states','episode','frame']})
 np.savez_compressed(RUN/'sampling_arrays.npz',**{k:arrays[k] for k in ['episode','frame','key','expert','recovery','eligible_train','eligible_validation','weights','source_pool']})
 stats={name:feature_stats(arrays[k],training&arrays['expert']) for name,k in [('action','actions'),('observation.state','states')]}
 for cam in CAMERAS:stats[cam]=dict(mean=[[[v]] for v in [.485,.456,.406]],std=[[[v]] for v in [.229,.224,.225]])
 write('train_stats.json',stats);write('split_manifest.json',dict(train_episodes=train,validation_episodes=val,source_records=records,old_manifest_sha256=sha(OLD/'split_manifest.json'),completion_evidence_sha256=audit['completion_evidence_sha256'],split_rule='heldout grid(1,1),(2,3); same seed/layout grouping across every source'))
 write('sampling_audit.json',audit);write('preparation_complete.json',dict(complete=True,episodes=len(records),manifest_sha256=sha(RUN/'split_manifest.json')));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
