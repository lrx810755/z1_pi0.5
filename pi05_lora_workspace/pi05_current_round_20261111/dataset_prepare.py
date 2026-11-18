"""Convert frozen raw demonstrations into pinned native PI0.5 training arrays."""
import hashlib
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from config import RUN,WORK
from rules import valid_anchors,recovery_mask
from label_rules import full_labels
from quality_decisions import accepted_reverse
from data_contracts import validate_timeline
FPS=10
HORIZON=30
OBJECTS=['apple','peach','cup','bottle']
CAMERAS=['observation.images.'+x for x in ['wrist','global','global_right']]
CROPS={CAMERAS[0]:(-140,0,640,640),CAMERAS[1]:(-80,0,640,640),CAMERAS[2]:(-80,0,640,640)}
sys.path.insert(0,'/home/lp531/robot/dataset_trajectory_review')
import review
import screen_fruit_motion as motion
from screen_cup_training import cup_command_labels
sys.path.insert(0,'/home/lp531/robot/apple_scale110_correction60_20261106')
from fragment_core import partial_labels

def write(name,data):
    p=RUN/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n');t.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p,default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default

def assert_pins(records):
    for r in records:
        for name,digest in r['source_hashes'].items():
            if sha(Path(r['source_path'])/name)!=digest:raise ValueError('Source changed: '+str(Path(r['source_path'])/name))

def quantile_stats(values,mask):
    x=values[mask].astype(np.float64)
    if not len(x) or not np.isfinite(x).all():raise ValueError('Invalid normalization values')
    return dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),q01=np.quantile(x,.01,axis=0).tolist(),q99=np.quantile(x,.99,axis=0).tolist(),count=[len(x)])

def old_cup_data():
    root=WORK/'pi05_cup_recovery513_lora_20261029'
    records=read(root/'split_manifest.json')['source_records']
    with np.load(root/'lowdim.npz') as data:actions=data['actions'].copy()
    return {str(Path(r['source_path']).resolve()):r for r in records},actions

def load_episode(root,old_records=None,old_actions=None):
    root=Path(root);rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
    summary=read(root/'episode_summary.json');meta=read(root/'correction.json',{});n=len(rows)
    fragment=summary.get('episode_goal') in ('apple_correction_grasp_lift','peach_correction_grasp_lift')
    if summary['fps']!=FPS or summary['frame_count']!=n or summary['dropped_frames']!=0:raise ValueError('Frame/fps mismatch: '+str(root))
    if fragment:
        if not (meta.get('accepted') is True and meta.get('recovery_complete') is True and summary.get('physical_grasp_ok') is True):raise ValueError('Unaccepted fragment: '+str(root))
    elif summary.get('task_success') is not True:raise ValueError('Unsuccessful full demonstration: '+str(root))
    obj={'beverage_bottle':'bottle'}.get(summary['target_object'],summary['target_object'])
    if obj not in OBJECTS:raise ValueError('Unexpected target '+obj)
    source_pool=meta.get('task',{}).get('source_pool') if isinstance(meta.get('task'),dict) else None
    source_pool=source_pool or meta.get('source_pool')
    if source_pool is not None and source_pool not in ('basic','correction'):
        raise ValueError('Unknown declared source pool: '+str(source_pool))
    # Retakes may carry correction.json for provenance while remaining basic demonstrations.
    pool=int(source_pool=='correction') if source_pool is not None else int(bool(meta))
    source=obj+('_correction' if pool else '_basic')
    states=np.asarray([r['observation.state'] for r in rows],np.float32)
    actions=np.asarray([r['action'] for r in rows],np.float32)
    times=np.asarray([r['simulation_time'] for r in rows])
    if states.shape!=(n,7) or actions.shape!=(n,7) or not np.isfinite(states).all() or not np.isfinite(actions).all():raise ValueError('Invalid joint arrays: '+str(root))
    if not np.isfinite(times).all() or not np.allclose(np.diff(times),.1,atol=1e-5,rtol=0):raise ValueError('Nonuniform timeline: '+str(root))
    if not all(r['frame_index']==i and r['action_stamp']<=r['simulation_time']+1e-5 and r['target_object']==summary['target_object'] and r['task']==summary['task'] for i,r in enumerate(rows)):raise ValueError('Causality/identity mismatch: '+str(root))
    timeline_audit=validate_timeline(rows)
    labels=read(root/'gripper_labels.json',{}).get('labels');repair=None
    label_source='saved_command_intent' if labels is not None else None
    basic_label_changed_frames=0
    old=(old_records or {}).get(str(root.resolve()))
    if old:
        for name,digest in old['source_hashes'].items():
            if sha(root/name)!=digest:raise ValueError('Historical source changed: '+str(root/name))
        cached=old_actions[old['dataset_from_index']:old['dataset_to_index']]
        if cached.shape!=actions.shape or not np.array_equal(cached[:,:6],actions[:,:6]):raise ValueError('Historical action alignment changed')
        if labels is not None and not np.array_equal(labels,cached[:,-1]):raise ValueError('Historical binary labels mismatch')
        labels=cached[:,-1]
        label_source='historical_verified_cup_intent'
    elif fragment and 'close_request_sim_time' not in meta:
        # Early accepted apple clips predate explicit command-event logging.
        # Keep their saved labels; never fabricate an event timestamp.
        label_meta=read(root/'gripper_labels.json',{})
        close=label_meta.get('close_command_frame')
        if not isinstance(close,int) or not 0<close<n or labels!=[int(i>=close) for i in range(n)]:
            raise ValueError('Legacy fragment saved labels/transition inconsistent: '+str(root))
    elif fragment:
        derived,_,_=partial_labels(rows,meta['close_request_sim_time'],[e['simulation_time'] for e in meta.get('gripper_command_events',[]) if e['state']=='open'])
        if labels is None:labels=derived
        elif len(labels)==n and np.array_equal(labels,derived):pass
        elif obj=='apple' and root.name=='episode_000055' and len(labels)==n-1 and labels==derived[:-1]:
            labels=derived;repair='Derived missing final label from explicit causal close event; raw unchanged'
        else:raise ValueError('Fragment label/event mismatch: '+str(root))
    elif labels is None:
        if pool:raise ValueError('Correction requires saved command-time gripper labels: '+str(root))
        labels,_,_=full_labels(rows)
        previous=cup_command_labels(rows) if obj in ('cup','bottle') else motion.command_labels(rows,root)[0]
        basic_label_changed_frames=int(np.sum(np.asarray(labels)!=previous))
        label_source='reconstructed_full_close_release_command_intent'
    if pool and meta.get('events'):
        declared=np.zeros(n,int)
        commands=sorted([e for e in meta['events'] if e.get('event')=='gripper_intent'],key=lambda e:e['time'])
        for event in commands:declared[times>=float(event['time'])-1e-7]=int(event['state']=='close')
        if not np.array_equal(labels,declared):raise ValueError('Saved correction labels do not match command events: '+str(root))
    if len(labels)!=n or not np.isin(labels,[0,1]).all():raise ValueError('Binary labels invalid: '+str(root))
    actions[:,-1]=labels
    expert=review.event_mask(meta,times)
    saved_masks=read(root/'training_mask.json',{})
    mask=saved_masks.get('expert_frame_mask')
    if mask is not None:
        if len(mask)!=n or not all(type(x) is bool for x in mask):raise ValueError('Expert mask mismatch')
        for h in [16,30]:
            declared=saved_masks.get(f'valid_action_horizon{h}_anchors')
            if declared is not None and not np.array_equal(declared,valid_anchors(np.asarray(mask,bool),h)):
                raise ValueError('Saved expert horizon mask mismatch: '+str(root))
        expert &= np.asarray(mask,bool)
    before_causal=int(expert.sum())
    expert &= review.event_mask(meta,np.asarray([r['action_stamp'] for r in rows],float))
    causal_extra_excluded=before_causal-int(expert.sum())
    valid=valid_anchors(expert,HORIZON)
    if not valid.any():raise ValueError('No complete expert window: '+str(root))
    key=np.zeros(n,bool)
    for t in np.flatnonzero(np.diff(labels)!=0)+1:key[max(0,t-8):min(n,t+9)]=True
    recovery,intervals=recovery_mask(rows,meta,expert)
    if fragment:recovery=expert.copy()
    pins={p.name:sha(p) for p in root.glob('*.json')}
    pins['frames.jsonl']=sha(root/'frames.jsonl')
    if (root/'physics_status.jsonl').exists():pins['physics_status.jsonl']=sha(root/'physics_status.jsonl')
    record=dict(source=source,object=obj,source_path=str(root.resolve()),frames=n,split='train',task=summary['task'],original_task=summary['task'],grid=summary['scene_randomization']['target_grid'],seed=summary['scene_randomization']['seed'],source_hashes=pins,kind=meta.get('task',{}).get('kind') if isinstance(meta.get('task'),dict) else None,fragment=fragment,label_repair=repair,label_source=label_source,timeline_audit=timeline_audit,basic_label_changed_frames=basic_label_changed_frames,causal_intervention_extra_excluded_frames=causal_extra_excluded,recovery_intervals=intervals)
    arrays=dict(states=states,actions=actions,expert=expert,key=key,recovery=recovery,valid=valid,source_pool=np.full(n,2*OBJECTS.index(obj)+pool))
    return record,arrays,rows,meta

def reverse_report(record,arrays,rows,meta,chain):
    points,rot=review.forward(arrays['states'][:,:6].astype(float),chain)
    points+=np.einsum('nij,j->ni',rot,np.array([.051,0,0]))
    times=np.asarray([r['simulation_time'] for r in rows]);phases=np.array([r.get('phase','UNKNOWN') for r in rows],object)
    for stages,start,end in [(meta.get('stages',[]),'start_sim_time','end_sim_time'),(meta.get('recovery_stages',[]),'start','end')]:
        for s in stages:
            if not all(k in s for k in (start,end,'stage')):continue
            m=(times>=s[start])&(times<s[end]);phases[m]=[p+'#'+s['stage'] for p in phases[m]]
    segments=motion.reverse_segments(points,phases,arrays['actions'][:,-1],arrays['expert'],times,FPS,record['fragment'],motion.Rules())
    maximum=max((s['max_reverse_excursion_m'] for s in segments),default=0.)
    return dict(path=record['source_path'],frames_sha256=record['source_hashes']['frames.jsonl'],max_reverse_excursion_m=maximum,pending=maximum>=.08)

def main():
    from pipeline import verify_frozen
    frozen=verify_frozen();old,old_actions=old_cup_data();parts=defaultdict(list);records=[];start=0;quality=[]
    chain=review.chain_from_urdf(review.DEFAULT_URDF)
    # Remove stale success before any conversion. Failures cannot authorize training.
    (RUN/'preparation_complete.json').unlink(missing_ok=True)
    # Screen metadata first so a quality failure does not decode millions of JPEGs.
    for item in frozen['sources']:
        r,a,rows,meta=load_episode(Path(item['path']),old,old_actions)
        report=reverse_report(r,a,rows,meta,chain)
        report['accepted_by_user']=accepted_reverse(RUN,report['path'],report['frames_sha256'])
        quality.append(report)
    write('final_reverse_screen.json',quality)
    pending=[r for r in quality if r['pending'] and not r.get('accepted_by_user',False)]
    if pending:raise ValueError(f'{len(pending)} 条仍达到反向8cm阈值；见 final_reverse_screen.json，处理后重新整理转换')
    from PIL import Image
    for item in frozen['sources']:
        root=Path(item['path']);r,a,rows,meta=load_episode(root,old,old_actions);n=len(rows);idx=len(records)
        r.update(episode_index=idx,dataset_from_index=start,dataset_to_index=start+n);records.append(r)
        from image_audit import audit_episode
        image_pins,_=audit_episode(dict(path=str(root),frames_sha256=r['source_hashes']['frames.jsonl']),RUN/'image_audit_cache')
        sizes=image_pins['image_sizes'];mtime=image_pins['image_mtime_ns'];ctime=image_pins['image_ctime_ns'];hashes=image_pins['image_hashes']
        a.update(episode=np.full(n,idx),frame=np.arange(n),image_sizes=sizes,image_mtime_ns=mtime,image_ctime_ns=ctime,image_hashes=hashes)
        for k,v in a.items():parts[k].append(v)
        start+=n
        if idx%10==0:print(f'转换 {idx+1}/{len(frozen["sources"])}，累计 {start} 帧',flush=True)
    write('final_reverse_screen.json',quality)
    pending=[r for r in quality if r['pending'] and not r.get('accepted_by_user',False)]
    if pending:raise ValueError(f'{len(pending)} 条仍达到反向8cm阈值；见 final_reverse_screen.json，处理后重新整理转换')
    arrays={k:np.concatenate(v) for k,v in parts.items()};eligible=arrays['valid'];weights=np.maximum(np.where(arrays['key'],1.5,1.),np.where(arrays['recovery'],3.,1.));weights[~eligible]=0
    for pool in range(8):
        if np.sum(eligible&(arrays['source_pool']==pool))<3:raise ValueError('Empty or too small object/pool: '+str(pool))
    # Only frames actually used as state or action targets contribute to statistics.
    effective=np.zeros(start,bool)
    for r in records:
        a,b=r['dataset_from_index'],r['dataset_to_index'];anchors=np.flatnonzero(eligible[a:b]);delta=np.zeros(b-a+1,int)
        np.add.at(delta,anchors,1);np.add.at(delta,anchors+HORIZON,-1);effective[a:b]=np.cumsum(delta[:-1])>0
    stats={name:quantile_stats(arrays[key],effective) for name,key in [('action','actions'),('observation.state','states')]}
    for cam in CAMERAS:stats[cam]=dict(mean=[[[.5]]]*3,std=[[[.5]]]*3,min=[[[0.]]]*3,max=[[[1.]]]*3)
    np.savez(RUN/'lowdim.npz',**{k:arrays[k] for k in ['actions','states','episode','frame']})
    np.savez_compressed(RUN/'image_pins.npz',**{k:arrays[k] for k in ['image_sizes','image_mtime_ns','image_ctime_ns','image_hashes']})
    np.savez_compressed(RUN/'sampling_arrays.npz',**{k:arrays[k] for k in ['episode','frame','expert','key','recovery','source_pool']},eligible_train=eligible,eligible_validation=np.zeros(start,bool),weights=weights,effective=effective)
    write('split_manifest.json',dict(train_episodes=list(range(len(records))),validation_episodes=[],source_records=records,split_rule='All selected episodes train; original task preserved'))
    write('train_stats.json',stats)
    write('sampling_audit.json',dict(episodes=len(records),frames=start,sources=dict(Counter(r['source'] for r in records)),training_windows=int(eligible.sum()),per_update_per_object=dict(basic=3,correction=1),microbatch=2,accumulation=8,transition_radius_frames=8,user_retained_reverse_episodes=sum(q['pending'] and q.get('accepted_by_user',False) for q in quality),basic_label_changed_frames=sum(r['basic_label_changed_frames'] for r in records),causal_intervention_extra_excluded_frames=sum(r['causal_intervention_extra_excluded_frames'] for r in records),label_repairs=[r['source_path'] for r in records if r['label_repair']]))
    assert_pins(records);verify_frozen()
    files=['lowdim.npz','image_pins.npz','sampling_arrays.npz','split_manifest.json','train_stats.json','sampling_audit.json','sources_frozen.json','final_reverse_screen.json']
    write('preparation_complete.json',dict(passed=True,fps=FPS,chunk_size=HORIZON,episodes=len(records),accepted_new=len(records),files={str(RUN/f):sha(RUN/f) for f in files}))

if __name__=='__main__':main()
