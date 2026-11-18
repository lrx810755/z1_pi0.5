"""Annotate all 211 episodes, validate fine-stage sampling, then activate its input config."""
from pathlib import Path
import csv,hashlib,json,sys,shutil
import numpy as np
import pyarrow.parquet as pq
from phases import annotate,NAMES,WEIGHTS
from recovery import apply_recovery_weight
W=Path(__file__).resolve().parent;WORK=W.parent;WINDOWS=WORK/'correction_windows_20261012'
sys.path.insert(0,str(WINDOWS))
from sampling import WindowSampler

def load(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def distributions(probabilities,absolute,ids,episode_lengths,frame):
    anchor=np.bincount(ids[absolute],weights=probabilities,minlength=len(NAMES))
    target=np.zeros(len(NAMES));valid_slots=0.
    for offset in range(15):
        valid=frame[absolute]+offset<episode_lengths[absolute]
        target+=np.bincount(ids[absolute[valid]+offset],weights=probabilities[valid],minlength=len(NAMES))
        valid_slots+=float(probabilities[valid].sum())
    return dict(anchor_mass={k:float(v) for k,v in zip(NAMES,anchor)},
                valid_action_slot_mass={k:float(v/valid_slots) for k,v in zip(NAMES,target)},
                expected_valid_slots_per_chunk=valid_slots,padded_slot_fraction=1-valid_slots/15)

def main():
    baseconfig=load(WINDOWS/'training_input.json');DATA=Path(baseconfig['dataset_root'])
    windows=load(WINDOWS/'mapped_windows.json')['records']
    old={r['episode_index']:r for r in load(WORK/'grip_stage_base_20261011/phase_manifest.json')}
    with np.load(WINDOWS/'merged_211_sampling_arrays.npz') as archive:a={k:archive[k] for k in archive.files}
    table=pq.read_table(DATA/'data').sort_by([('index','ascending')])
    ep=table['episode_index'].to_numpy();fi=table['frame_index'].to_numpy()
    np.testing.assert_array_equal(ep,a['episode']);np.testing.assert_array_equal(fi,a['frame'])
    protected={str(p):sha(p) for folder in [DATA/'data'] for p in folder.rglob('*.parquet')}
    protected[str(DATA/'meta/stats.json')]=sha(DATA/'meta/stats.json')
    ids=np.zeros(len(ep),dtype=np.int8);weights=np.ones(len(ep));lengths=np.zeros(len(ep),dtype=np.int64)
    records=[];missing_hold=[];recovery_mask=np.zeros(len(ep),dtype=bool)
    for r in windows:
        index=r['episode_index'];source=old[index] if index<192 else r
        path=Path(source['source_path'])/'frames.jsonl'
        assert sha(path)==source['source_frames_sha256'],path
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        close=source['close_frame'];opening=source.get('open_frame')
        labels,weight,event=annotate(rows,close,opening)
        recovery=np.zeros(len(rows),dtype=bool)
        if index>=160:
            correction_path=path.parent/'correction.json'
            correction_meta=load(correction_path)
            assert correction_meta['accepted'] and correction_meta['recovery_complete']
            weight,recovery=apply_recovery_weight(rows,correction_meta['stages'],close,weight)
            event.update(recovery_metadata_sha256=sha(correction_path),recovery_frames=int(recovery.sum()),
                         recovery_intervals=correction_meta['stages'])
        mask=ep==index;assert mask.sum()==len(rows)==r['frame_count']
        assert np.array_equal(fi[mask],np.arange(len(rows)))
        ids[mask]=labels;weights[mask]=weight;lengths[mask]=len(rows)
        recovery_mask[mask]=recovery
        if index>=160:assert event['stable_confirmed']==r['stable_confirmed_frame']
        if event['hold_start'] is None:missing_hold.append(index)
        entry=dict(episode_index=index,source=r['source'],split=r['split'],source_path=str(path.parent),
                   source_frames_sha256=sha(path),frame_count=len(rows),**event,
                   phase_frame_counts={k:int((labels==i).sum()) for i,k in enumerate(NAMES)})
        records.append(entry)
    assert len(records)==211
    arrays=W/'fine_sampling_arrays.npz';np.savez(arrays,**dict(a,phase_weight=weights,phase_id=ids,recovery=recovery_mask))
    counts=[r['frame_count'] for r in windows];ends=np.cumsum(counts);starts=np.r_[0,ends[:-1]]
    fine=WindowSampler(starts,ends,shuffle=True,seed=20261012,eligibility_path=arrays)
    previous=WindowSampler(starts,ends,shuffle=True,seed=20261012,eligibility_path=W/'before_recovery_restore/fine_sampling_arrays.npz')
    np.testing.assert_array_equal(fine._absolute_indices,previous._absolute_indices)
    absolute=fine._absolute_indices;p=fine._probabilities.numpy();previous_p=previous._probabilities.numpy()
    correction=a['correction'][absolute]
    recovery_selected=recovery_mask[absolute]
    assert np.all(weights[recovery_mask]==2.) and weights.max()==2.
    assert not recovery_mask[ep<160].any()
    recovery_before=float(previous_p[recovery_selected].sum());recovery_after=float(p[recovery_selected].sum())
    assert recovery_after>recovery_before
    assert abs(p[correction].sum()-.3)<1e-12 and abs(p[~correction].sum()-.7)<1e-12
    drawn=np.asarray(list(fine));assert a['eligible_train_anchor'][drawn].all()
    held={r['episode_index'] for r in windows if r['split']=='validation'};assert not set(ep[drawn])&held
    resume=WindowSampler(starts,ends,shuffle=True,seed=20261012,eligibility_path=arrays)
    resume.load_state_dict(dict(epoch=0,start_index=123));np.testing.assert_array_equal(np.asarray(list(resume)),drawn[123:])
    expected=distributions(p,absolute,ids,lengths,fi)
    empirical=distributions(np.full(len(drawn),1/len(drawn)),drawn,ids,lengths,fi)
    assert max(abs(expected['anchor_mass'][k]-empirical['anchor_mass'][k]) for k in NAMES)<.015
    pools={}
    for name,flag in [('original',False),('correction',True)]:
        m=correction==flag
        pools[name]=distributions(p[m]/p[m].sum(),absolute[m],ids,lengths,fi)
    audit=dict(passed=True,episodes_annotated=211,train_episodes=203,validation_episodes=8,
               stage_names=NAMES,stage_weights={k:float(v) for k,v in zip(NAMES,WEIGHTS)},
               anchor_weight_rule='max(fine stage weight, 2 for logged pre-close recovery); normalize within 70/30 source pools',
               recovery_weight=2,overlap_rule='maximum',
               recovery_train_anchors=int(recovery_selected.sum()),
               recovery_anchor_mass_before=recovery_before,recovery_anchor_mass_after=recovery_after,
               loss_weight=1,chunk_size=15,eligible_train_anchors=len(absolute),
               episodes_without_detectable_prelift_hold=missing_hold,
               stage_training_anchor_counts={k:int((ids[absolute]==i).sum()) for i,k in enumerate(NAMES)},
               previous=distributions(previous_p,absolute,ids,lengths,fi),current=expected,
               empirical_draws=empirical,within_pool=pools,
               source_pool_mass=dict(original=float(p[~correction].sum()),correction=float(p[correction].sum())),
               all_source_frame_hashes_match=True,eligibility_and_splits_unchanged=True,
               actual_sampler_draws_verified=len(drawn),deterministic_resume_verified=True,
               data_and_normalization_unchanged=True,training_started=False)
    assert all(sha(Path(path))==h for path,h in protected.items())
    save(W/'phase_manifest.json',records);save(W/'sampling_audit.json',audit)
    with (W/'phase_boundaries.csv').open('w') as out:
        keys=['episode_index','source','split','preclose_start','close_start','hold_start','lift_start','stable_confirmed','opening','hold_frames','source_path']
        writer=csv.DictWriter(out,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    shutil.copy2(arrays,DATA/'meta/fine_grip_sampling_arrays.npz')
    shutil.copy2(W/'phase_manifest.json',DATA/'meta/fine_grip_phase_manifest.json')
    config=dict(baseconfig,sampling_scheme='fine_grip_with_recovery_20261012',
                sampler_kwargs=dict(eligibility_path=str(DATA/'meta/fine_grip_sampling_arrays.npz')),
                phase_manifest=str(DATA/'meta/fine_grip_phase_manifest.json'),
                stage_weights=audit['stage_weights'],anchor_weight_rule=audit['anchor_weight_rule'],loss_weight=1,
                recovery_weight=2,overlap_rule='maximum')
    if not (W/'previous_training_input.json').exists():save(W/'previous_training_input.json',baseconfig)
    save(W/'training_input.json',config)
    # Activate only after complete annotation and sampler verification.
    save(WINDOWS/'training_input.json',config);save(DATA/'meta/training_sampling_config.json',config)
    print(json.dumps({k:audit[k] for k in ['passed','episodes_annotated','eligible_train_anchors','recovery_train_anchors','recovery_anchor_mass_before','recovery_anchor_mass_after','source_pool_mass']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
