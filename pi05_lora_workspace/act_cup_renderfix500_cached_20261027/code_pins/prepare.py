"""Freeze 500 accepted render-fixed episodes; audit masks, labels, references and split."""
import argparse,hashlib,json,math,shutil,time
from pathlib import Path
import numpy as np
from rules import (PLAN_HASH,EXPECTED_CATEGORIES,split_for_grid,validate_collection,validate_reference,
                   correction_labels,valid_anchors,feature_stats,full_labels)
from weights import source_balanced_weights
RUN=Path(__file__).resolve().parent
DATA=RUN/'cache'
BASE=Path('/home/lp531/smolvla_data/z1_tabletop_raw_cup_basic_4x5x20_renderfix_20261026')
CORR=Path('/home/lp531/smolvla_data/z1_cup_correction100_full_renderfix_20261027')
PLAN=Path('/home/lp531/z1_queue_logs/cup_correction100_full_renderfix_20261027/collection_plan.json')
CAMERAS=['observation.images.'+x for x in ['wrist','global','global_right']]
CROPS={CAMERAS[0]:(-140,0,640,640),CAMERAS[1]:(-80,0,640,640),CAMERAS[2]:(-80,0,640,640)}

def write(name,value):
 p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str));t.replace(p)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def required_source_hashes(path,source):
 if source not in {'basic','correction'}:raise ValueError('Unknown source kind')
 names=['frames.jsonl','episode_summary.json','episode_start.json']
 if source=='correction':names+=['correction.json','gripper_labels.json','training_mask.json','physics_status.jsonl']
 for name in names:
  p=Path(path)/name
  if not p.is_file() or p.stat().st_size==0:raise ValueError(f'Missing or empty required source evidence: {p}')
 return {name:sha(Path(path)/name) for name in names}

def validate_physics(path,meta):
 expected=meta.get('dynamic_samples')
 if type(expected) is not int or expected<1:raise ValueError('Invalid physics dynamic_samples count')
 count=0;previous=-math.inf
 try:
  with Path(path).open() as stream:
   for line in stream:
    sample=json.loads(line)
    if not isinstance(sample,dict) or not isinstance(sample.get('status'),dict):raise ValueError('Malformed physics sample/status structure')
    status=sample['status'];stamp=sample['simulation_time']
    if isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or stamp<previous:raise ValueError('Invalid physics simulation_time')
    previous=stamp
    for key,value in [('frozen',False),('target_collisions_enabled',True),('left_finger_touching_table',False),('right_finger_touching_table',False)]:
     if status.get(key) is not value:raise ValueError(f'Unsafe or missing physics status {key}: {path}, sample {count+1}')
    count+=1
 except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:raise ValueError(f'Malformed physics evidence: {path}: {exc}') from exc
 if count!=expected:raise ValueError(f'Incomplete physics evidence: {path}: {count} samples, expected {expected}')
 return count

def collection_gate():
 basic=sorted(BASE.glob('episode_[0-9]*'));corrections=sorted(CORR.glob('episode_[0-9]*'));short=[]
 for p in corrections:
  c=json.loads((p/'correction.json').read_text())
  short.append(dict(kind=c['task']['kind'],task_id=c['task_id'],plan_hash=c['plan_hash']))
 validate_collection(len(basic),short)
 return basic,corrections

def assert_pins(records):
 for r in records:
  root=Path(r['source_path'])
  for name,digest in r['source_hashes'].items():
   if sha(root/name)!=digest:raise ValueError(f'Source changed after pin: {root/name}')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
 basic,corrections=collection_gate() # hard gate precedes cache allocation or metadata freeze
 if (RUN/'preparation_complete.json').exists() and not args.audit_only:
  assert_pins(json.loads((RUN/'split_manifest.json').read_text())['source_records']);print('Frozen preparation verified',flush=True);return
 started=time.monotonic();records=[];parts={k:[] for k in ['actions','states','episode','frame','expert','valid','key','source_pool']};start=0
 basic_by_path={};plan=json.loads(PLAN.read_text());plan_tasks=plan['tasks']
 plan_ids={t['id'] for t in plan_tasks}
 if plan.get('hash')!=PLAN_HASH:raise ValueError('Plan metadata hash mismatch')
 if len(plan_ids)!=100:raise ValueError('Plan must contain100 unique tasks')
 for source,paths in [('basic',basic),('correction',corrections)]:
  for path in paths:
   hashes=required_source_hashes(path,source)
   summary=json.loads((path/'episode_summary.json').read_text())
   for k in ['physical_grasp_ok','task_success','release_sequence_complete','placement_stable']:
    if not summary[k]:raise ValueError(f'{path}: unsuccessful {k}')
   if summary['target_object']!='cup' or summary['dropped_frames']!=0:raise ValueError('Wrong target or dropped frames')
   rows=[json.loads(line) for line in (path/'frames.jsonl').read_text().splitlines()];n=len(rows)
   if n!=summary['frame_count'] or n<16:raise ValueError('Invalid frame count')
   grid=summary['scene_randomization']['target_grid'];seed=summary['scene_randomization']['seed'];split=split_for_grid(grid)
   meta={}
   if source=='basic':
    labels,close,release=full_labels(rows);expert=np.ones(n,bool);valid=valid_anchors(expert);transitions=[close,release]
    basic_by_path[str(path.resolve())]=(grid,seed)
   else:
    meta=json.loads((path/'correction.json').read_text())
    validate_physics(path/'physics_status.jsonl',meta)
    labels,expert,valid=correction_labels(json.loads((path/'gripper_labels.json').read_text()),json.loads((path/'training_mask.json').read_text()),n)
    for k in ['accepted','recovery_complete']:
     if not meta[k]:raise ValueError(f'{path}: {k} failed')
    if meta['table_contact'] or meta['dynamic_violations']:raise ValueError('Unsafe correction')
    final=meta['final_status']
    for k in ['task_success','physical_grasp_ok','released','placement_stable','release_sequence_complete','target_collisions_enabled']:
     if not final[k]:raise ValueError(f'Final status failed: {k}')
    if final['frozen'] or final['holding_observed'] or abs(meta['measured_preclose_roll_deg']+90)>=3:raise ValueError('Invalid final physics or orientation')
    if meta['split']!=split or meta['task']['split']!=split or meta['task_id'] not in plan_ids:raise ValueError('Correction plan/split mismatch')
    if list(meta['task']['grid'])!=list(grid):raise ValueError('Correction grid mismatch')
    ref=str(Path(meta['task']['reference_basic_episode']).resolve())
    if ref not in basic_by_path:raise ValueError('Unknown basic reference')
    validate_reference(grid,basic_by_path[ref][0],seed,basic_by_path[ref][1])
    transitions=(np.flatnonzero(np.diff(labels)!=0)+1).tolist()
    if not transitions or labels[0]!=0 or labels[-1]!=0:raise ValueError('Incomplete correction gripper intent')
   a=np.asarray([r['action'] for r in rows],np.float32);s=np.asarray([r['observation.state'] for r in rows],np.float32)
   if a.shape!=(n,7) or s.shape!=(n,7) or not np.isfinite(a).all() or not np.isfinite(s).all():raise ValueError('Invalid lowdim values')
   a[:,-1]=labels;key=np.zeros(n,bool)
   for t in transitions:key[max(0,t-8):min(n,t+8)]=True
   for i,row in enumerate(rows):
    if row['frame_index']!=i or row['target_object']!='cup':raise ValueError('Frame alignment invalid')
    for cam in CAMERAS:
     image=path/row[cam]
     if not image.is_file() or image.stat().st_size==0 or not image.resolve().is_relative_to(path.resolve()):raise ValueError(f'Missing/unsafe camera path {image}')
   idx=len(records)
   record=dict(episode_index=idx,source=source,source_path=str(path),frames=n,grid=grid,seed=seed,split=split,
      task=rows[0]['task'],task_id=meta.get('task_id'),kind=meta.get('task',{}).get('kind'),transitions=transitions,
      dataset_from_index=start,dataset_to_index=start+n,source_hashes=hashes,source_frames_sha256=hashes['frames.jsonl'],
      expert_frames=int(expert.sum()),eligible_windows=int(valid.sum()),reference_basic_episode=meta.get('task',{}).get('reference_basic_episode'))
   records.append(record)
   for k,v in dict(actions=a,states=s,episode=np.full(n,idx,np.int32),frame=np.arange(n,dtype=np.int32),expert=expert,valid=valid,key=key,source_pool=np.full(n,int(source=='correction'),np.uint8)).items():parts[k].append(v)
   start+=n
   if len(records)%10==0:
    status=dict(episodes=len(records),target=500,frames=start,elapsed_s=time.monotonic()-started);write('prepare_progress.json',status);print(json.dumps(status),flush=True)
 arrays={k:np.concatenate(v) for k,v in parts.items()};ep=arrays['episode'];train=[r['episode_index'] for r in records if r['split']=='train'];val=[r['episode_index'] for r in records if r['split']=='validation']
 if len(train)!=450 or len(val)!=50:raise ValueError(f'Expected450/50 spatial split, got{len(train)}/{len(val)}')
 for field in ['seed','source_frames_sha256']:
  if {r[field] for r in records if r['split']=='train'}&{r[field] for r in records if r['split']=='validation'}:raise ValueError(f'Train/validation {field} overlap')
 train_mask=np.isin(ep,train);eligible=train_mask&arrays['valid'];valmask=~train_mask&arrays['valid'];w=source_balanced_weights(eligible,arrays['key'],arrays['source_pool'])
 report=dict(episodes=500,frames=start,train_episodes=450,validation_episodes=50,source_counts={'basic':400,'correction':100},
  correction_categories=EXPECTED_CATEGORIES,training_windows=int(eligible.sum()),validation_windows=int(valmask.sum()),
  source_mass={'basic':float(w[arrays['source_pool']==0].sum()),'correction':float(w[arrays['source_pool']==1].sum())},
  excluded_intervention_frames=int((~arrays['expert']).sum()),phase_multiplier=1.5,phase_radius_frames=8,
  crop_cache_bytes=start*3*3*224*224*4,plan_hash=PLAN_HASH,plan_file_sha256=sha(PLAN),train_stats_rule='training episodes AND expert_frame_mask; no heldout or deliberate intervention frames')
 write('source_audit.json',report)
 if args.audit_only:print(json.dumps(report,indent=2));return
 DATA.mkdir(exist_ok=True)
 for r in records:
  pin=RUN/'source_pins'/f'episode_{r["episode_index"]:06d}';pin.mkdir(parents=True,exist_ok=True)
  for name in r['source_hashes']:shutil.copyfile(Path(r['source_path'])/name,pin/name)
 shutil.copyfile(PLAN,RUN/'source_pins'/'collection_plan.json')
 np.savez_compressed(RUN/'sampling_arrays.npz',episode=ep,frame=arrays['frame'],key=arrays['key'],expert=arrays['expert'],eligible_train=eligible,eligible_validation=valmask,weights=w,source_pool=arrays['source_pool'])
 np.savez(DATA/'lowdim.npz',**{k:arrays[k] for k in ['actions','states','episode','frame']})
 stats={name:feature_stats(arrays[k],train_mask&arrays['expert']) for name,k in [('action','actions'),('observation.state','states')]}
 for cam in CAMERAS:stats[cam]=dict(mean=[[[v]] for v in [.485,.456,.406]],std=[[[v]] for v in [.229,.224,.225]])
 write('train_stats.json',stats);write('split_manifest.json',dict(train_episodes=train,validation_episodes=val,source_records=records,split_rule='heldout grid cells(1,1),(2,3), every source; reference scenes stay in same split',plan_hash=PLAN_HASH,plan_file_sha256=sha(PLAN)))
 write('sampling_audit.json',report);write('preparation_complete.json',dict(complete=True,episodes=500,frames=start,manifest_sha256=sha(RUN/'split_manifest.json')))
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
