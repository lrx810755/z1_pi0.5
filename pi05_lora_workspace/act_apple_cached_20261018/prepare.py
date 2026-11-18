"""Frozen apple source manifest, spatial split and command-aligned ACT windows."""
import json,hashlib
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/act_apple_rgb224_cache_20261018')
BASE=Path('/home/lp531/smolvla_data/z1_tabletop_raw_apple_preplanned_three_camera_20261016')
CORR=Path('/home/lp531/smolvla_data/z1_apple_correction_fragments_20261016')
CAMERAS=['observation.images.'+x for x in ['wrist','global','global_right']]
CROPS={CAMERAS[0]:(-140,0,640,640),CAMERAS[1]:(-80,0,640,640),CAMERAS[2]:(-80,0,640,640)}
def write(name,value):
 p=RUN/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str));t.replace(p)
def full_labels(rows):
 a=np.asarray([r['action'][-1] for r in rows]);strong=np.flatnonzero(a>-.75)
 if not len(strong):raise ValueError('no confirmed closing ramp')
 held=int(strong[0]);before=np.flatnonzero(a[:held]<=-.895)
 if not len(before):raise ValueError('no open command before close')
 close=int(before[-1]+1);fully=np.flatnonzero((np.arange(len(a))>held)&(a<=-.899))
 if not len(fully):raise ValueError('no final release')
 end=int(fully[0]);start=end
 while start>held and a[start]<a[start-1]-1e-4:start-=1
 release=start+1
 if not(close<held<release<=end) or np.max(a[end:])>=-.88:raise ValueError('ambiguous opening')
 labels=np.zeros(len(rows),np.float32);labels[close:release]=1
 return labels,close,release

def main():
 if (RUN/'preparation_complete.json').exists():print('Frozen preparation already complete');return
 DATA.mkdir(exist_ok=True)
 records=[];actions=[];states=[];episode=[];frame=[];keys=[];sources=[];start=0
 for source,root in [('original',BASE),('correction',CORR)]:
  for path in sorted(root.glob('episode_[0-9]*')):
   summary=json.loads((path/'episode_summary.json').read_text())
   assert summary['target_object']=='apple' and summary['physical_grasp_ok']
   assert summary.get('task_success') if source=='original' else summary.get('fragment_success')
   rows=[json.loads(l) for l in (path/'frames.jsonl').read_text().splitlines()];n=len(rows)
   assert n==summary['frame_count'] and summary['dropped_frames']==0 and n>=16
   if source=='original':labels,close,release=full_labels(rows);meta={}
   else:
    meta=json.loads((path/'correction.json').read_text());lab=json.loads((path/'gripper_labels.json').read_text());labels=np.array(lab['labels'],np.float32);close=lab['close_command_frame'];release=None
    assert meta['accepted'] and meta['recovery_complete'] and not meta['table_contact']
    assert len(labels)<=n and np.array_equal(labels,np.arange(len(labels))>=close)
    assert all(row['action'][-1]>-.88 for row in rows[len(labels):]),'extra tail must remain closed'
    labels=(np.arange(n)>=close).astype(np.float32)
   a=np.asarray([x['action'] for x in rows],np.float32);s=np.asarray([x['observation.state'] for x in rows],np.float32)
   assert a.shape==s.shape==(n,7) and np.isfinite(a).all() and np.isfinite(s).all()
   a[:,-1]=labels;grid=summary['scene_randomization']['target_grid'];split='validation' if tuple(grid) in {(1,1),(2,3)} else 'train'
   if source=='correction':assert meta['split']==split
   key=np.zeros(n,bool);key[max(0,close-20):close+1]=True
   for stage in meta.get('stages',[]):
    for i,row in enumerate(rows[:close]):
     if stage['start_sim_time']<=row['simulation_time']<=stage['end_sim_time']:key[i]=True
   for row in rows:
    for cam in CAMERAS:assert (path/row[cam]).is_file(),path/row[cam]
   idx=len(records)
   records.append(dict(episode_index=idx,source=source,source_path=str(path),frames=n,close_frame=close,open_frame=release,grid=grid,split=split,seed=summary['scene_randomization']['seed'],source_frames_sha256=hashlib.sha256((path/'frames.jsonl').read_bytes()).hexdigest(),dataset_from_index=start,dataset_to_index=start+n,task=rows[0]['task'],task_id=meta.get('task_id'),label_tail_extended=(n-len(lab['labels']) if source=='correction' else 0)))
   actions.extend(a);states.extend(s);episode.extend([idx]*n);frame.extend(range(n));keys.extend(key);sources.extend([source=='correction']*n);start+=n
 train=[r['episode_index'] for r in records if r['split']=='train'];val=[r['episode_index'] for r in records if r['split']=='validation']
 assert not {r['source_frames_sha256'] for r in records if r['split']=='train'}&{r['source_frames_sha256'] for r in records if r['split']=='validation'}
 a=np.asarray(actions);s=np.asarray(states);ep=np.asarray(episode);f=np.asarray(frame);key=np.asarray(keys);source=np.asarray(sources)
 train_mask=np.isin(ep,train);valid=np.array([i+15<records[e]['frames'] for e,i in zip(ep,f)]);eligible=train_mask&valid;valmask=~train_mask&valid;w=np.zeros(len(ep),np.float64)
 for pool,mass in [(False,.7),(True,.3)]:
  mask=eligible&(source==pool);v=np.where(key[mask],2.,1.);w[mask]=mass*v/v.sum()
 np.savez_compressed(RUN/'sampling_arrays.npz',episode=ep,frame=f,key=key,eligible_train=eligible,eligible_validation=valmask,weights=w,source_correction=source)
 np.savez(DATA/'lowdim.npz',actions=a,states=s,episode=ep,frame=f)
 stats={}
 for name,values in [('action',a),('observation.state',s)]:
  x=values[train_mask].astype(np.float64);stats[name]=dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),count=[len(x)])
 for cam in CAMERAS:stats[cam]=dict(mean=[[[x]] for x in [.485,.456,.406]],std=[[[x]] for x in [.229,.224,.225]])
 write('train_stats.json',stats)
 write('split_manifest.json',dict(train_episodes=train,validation_episodes=val,source_records=records,split_rule='grid-heldout (1,1),(2,3) across both sources',source_hash_overlap=False))
 write('sampling_audit.json',dict(episodes=len(records),original=sum(r['source']=='original' for r in records),correction=sum(r['source']=='correction' for r in records),frames=len(ep),train_episodes=len(train),validation_episodes=len(val),original_mass=float(w[~source].sum()),correction_mass=float(w[source].sum()),training_windows=int(eligible.sum()),validation_windows=int(valmask.sum()),key_weight=2,other_weight=1,close_first_frame_included=True,rule='2s preclose through first close UNION recovery stages; full 16-action chunks only',binary_action_only=True,state_gripper_continuous=True))
 write('preparation_complete.json',dict(complete=True,episodes=len(records),frames=len(ep),cache=str(DATA)))
 print((RUN/'sampling_audit.json').read_text(),flush=True)
if __name__=='__main__':main()
