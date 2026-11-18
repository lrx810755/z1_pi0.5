"""Map frozen windows to the converted 211-episode dataset, then verify actual draws."""
from pathlib import Path
import hashlib,json,sys,csv,shutil
import numpy as np
import pyarrow.parquet as pq
from mapping import map_records
from windows import anchor_mask
from sampling import WindowSampler
W=Path(__file__).resolve().parent;WORK=W.parent
DATA=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012')
BASE=Path('/home/lp531/smolvla_data/z1_bottle_gentle_correction_20261010')
sys.path.insert(0,str(WORK/'binary_gripper_20261007'))
from convert import label_episode
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/scripts/correction_fragments_20261012')
from fragment_core import partial_labels

def load(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=load(W/'windows.json');old=load(WORK/'gentle_correction_20261010/snapshot.json')
    new=load(WORK/'correction_fragments_20261012/snapshot.json')
    assert len(old)==32 and len(new)==19
    table=pq.read_table(DATA/'data').sort_by([('index','ascending')]);base=pq.read_table(BASE/'data').sort_by([('index','ascending')])
    ep=table['episode_index'].to_numpy();fi=table['frame_index'].to_numpy();index=table['index'].to_numpy()
    assert len(table)==47135 and np.array_equal(index,np.arange(len(table))) and set(ep)==set(range(211))
    for key in ['action','observation.state']:assert table[key].to_pylist()[:len(base)]==base[key].to_pylist()
    assert sha(DATA/'meta/stats.json')==sha(BASE/'meta/stats.json')
    sources={e['source']:e['episode_index'] for e in old}
    sources.update({e['source']:192+i for i,e in enumerate(new)})
    bounds={i:(int(np.flatnonzero(ep==i)[0]),int((ep==i).sum())) for i in range(211)}
    for i,(start,n) in bounds.items():assert np.array_equal(fi[start:start+n],np.arange(n))
    records=map_records(manifest['records'],sources,bounds)
    assert [r['episode_index'] for r in records]==list(range(211))
    acts=np.asarray(table['action'].to_pylist(),dtype=np.float32)
    states=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
    phase=np.ones(len(table),dtype=np.float64)
    with np.load(WORK/'grip_stage_base_20261011/sampling_arrays.npz') as a:
        np.testing.assert_array_equal(a['episode'],ep[:len(base)])
        phase[:len(base)]=a['phase_weight']
    for r in records[160:]:
        p=Path(r['source_path']);assert sha(p/'frames.jsonl')==r['source_frames_sha256']
        rows=[json.loads(x) for x in (p/'frames.jsonl').read_text().splitlines()]
        labels=label_episode(rows)[0] if r['source']=='old_correction' else partial_labels(rows)[0]
        start=r['dataset_absolute_start'];stop=start+r['frame_count']
        expected=np.asarray([row['action'] for row in rows],dtype=np.float32);expected[:,-1]=labels
        np.testing.assert_array_equal(acts[start:stop],expected)
        np.testing.assert_array_equal(states[start:stop],np.asarray([row['observation.state'] for row in rows],dtype=np.float32))
        for local in range(r['anchor_start'],r['anchor_end_inclusive']+1):
            assert local+14<=r['frame_end_inclusive'] and len(labels[local:local+15])==15
            for j in range(local,local+15):
                if j>=r['close_frame']:assert labels[j]==1
        if r['source']=='new_fragment':
            close=r['close_frame'];stable=r['stable_confirmed_frame']
            phase[start+max(0,close-10):start+close]=1.5
            phase[start+close:start+stable+1]=2.
    eligible=anchor_mask(ep,fi,records)
    validation=anchor_mask(ep,fi,[dict(r,split='train' if r['split']=='validation' else 'validation') for r in records])
    held=[r['episode_index'] for r in records if r['split']=='validation']
    assert len(held)==8 and not np.isin(ep[eligible],held).any() and not (eligible&validation).any()
    assert eligible[ep<160].all()
    arrays=W/'merged_211_sampling_arrays.npz'
    np.savez(arrays,episode=ep,frame=fi,correction=ep>=160,phase_weight=phase,
             eligible_train_anchor=eligible,eligible_validation_anchor=validation)
    mapped=dict(manifest,dataset=str(DATA),records=records,new_fragments_converted_or_merged=True,
                episode_ids_are_actual=True,source_manifest_sha256=sha(W/'windows.json'))
    save(W/'mapped_windows.json',mapped)
    keys=['episode_index','source','task_id','split','frame_count','frame_start','frame_end_inclusive','anchor_start','anchor_end_inclusive','dataset_absolute_start','absolute_anchor_start','absolute_anchor_end_inclusive','source_path']
    with (W/'mapped_windows.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records[160:])
    dataset_arrays=DATA/'meta/sampling_eligibility.npz'
    shutil.copy2(arrays,dataset_arrays)
    shutil.copy2(W/'mapped_windows.json',DATA/'meta/correction_sampling_windows.json')
    starts=[bounds[i][0] for i in range(211)];ends=[bounds[i][0]+bounds[i][1] for i in range(211)]
    sampler=WindowSampler(starts,ends,shuffle=True,seed=20261012,eligibility_path=arrays)
    drawn=np.asarray(list(sampler));assert eligible[drawn].all()
    correction=ep[sampler._absolute_indices]>=160
    correction_mass=float(sampler._probabilities[correction].sum());assert abs(correction_mass-.3)<1e-12
    resume=WindowSampler(starts,ends,shuffle=True,seed=20261012,eligibility_path=arrays)
    resume.load_state_dict(dict(epoch=0,start_index=123));np.testing.assert_array_equal(np.asarray(list(resume)),drawn[123:])
    for r in records[160:]:
        assert not np.any(eligible[(ep==r['episode_index'])&((fi<r['anchor_start'])|(fi>r['anchor_end_inclusive']))])
    info=load(DATA/'meta/info.json');assert info['total_episodes']==211 and info['total_frames']==47135 and info['fps']==10
    config=dict(dataset_root=str(DATA),window_manifest=str(W/'mapped_windows.json'),chunk_size=15,
                sampler_source=str(W/'sampling.py'),sampler_class='WindowSampler',
                sampler_kwargs=dict(eligibility_path=str(dataset_arrays)),validation_episodes=held,
                train_episode_count=203,validation_episode_count=8,training_started=False)
    save(W/'training_input.json',config)
    save(DATA/'meta/training_sampling_config.json',config)
    audit=dict(passed=True,merged_dataset=str(DATA),episodes=211,train_episodes=203,validation_episodes=held,
               total_frames=len(table),original_episodes=160,old_correction_episodes=32,new_fragment_episodes=19,
               correction_train_episodes=43,correction_validation_episodes=8,
               source_actions_states_verified_for_all_51_corrections=True,old_dataset_actions_states_unchanged=True,
               normalization_preserved=True,all_source_frame_hashes_match=True,
               actual_dataset_indices_mapped=True,all_15_step_correction_chunks_complete=True,
               no_reopening_labels_in_correction_chunks=True,validation_leak=False,
               eligible_train_anchors=int(eligible.sum()),eligible_correction_train_anchors=int(eligible[ep>=160].sum()),
               eligible_validation_anchors=int(validation.sum()),actual_sampler_draws_verified=len(drawn),
               correction_pool_mass=correction_mass,deterministic_resume_verified=True,
               sampling_arrays_sha256=sha(arrays),training_started=False)
    save(W/'mapping_audit.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
