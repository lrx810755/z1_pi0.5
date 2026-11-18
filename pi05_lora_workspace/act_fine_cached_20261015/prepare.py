"""Preserve ACT split/windows/statistics and apply the existing fine phases."""
from pathlib import Path
import json,hashlib
import numpy as np
RUN=Path(__file__).resolve().parent
DATA=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012')
BASE=RUN.parent/'act_gentle_cached_20261015'
NAMES=['other','preclose','closing','prelift_hold','initial_lift','stable_hold']
STAGE_WEIGHTS=np.array([1.,1.5,2.,2.,1.5,1.])
def write(name,value):
    p=RUN/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str));t.replace(p)
def main():
    from sampling import weighted_probabilities
    with np.load(BASE/'sampling_arrays.npz') as old:arrays={k:old[k].copy() for k in old.files}
    fine=np.load(DATA/'meta/fine_grip_sampling_arrays.npz')
    for k in ['episode','frame']:np.testing.assert_array_equal(arrays[k],fine[k])
    phase=fine['phase_id'].copy();recovery=fine['recovery'].copy()
    expected=np.maximum(STAGE_WEIGHTS[phase],np.where(recovery,2.,1.))
    np.testing.assert_array_equal(expected,fine['phase_weight'])
    arrays['phase_id']=phase;arrays['recovery']=recovery;arrays['phase_weight']=expected
    arrays['key']=expected>1
    arrays['weights']=weighted_probabilities(arrays['episode'],arrays['eligible_train'],expected)
    split=json.loads((RUN/'split_manifest.json').read_text());starts=[]
    for r in split['source_records']:
        if r['episode_index'] not in split['train_episodes']:continue
        ids=np.flatnonzero((arrays['episode']==r['episode_index'])&(arrays['frame']==r['close_frame']))
        assert len(ids)==1 and arrays['eligible_train'][ids[0]]
        i=int(ids[0]);assert phase[i]==2 and expected[i]==2
        starts.append(dict(episode=r['episode_index'],frame=r['close_frame'],phase='closing',phase_weight=float(expected[i]),sampling_probability=float(arrays['weights'][i])))
    assert len(starts)==190
    assert not np.isin(arrays['episode'][arrays['weights']>0],split['validation_episodes']).any()
    for k in ['eligible_train','eligible_validation']:
        np.testing.assert_array_equal(arrays[k],np.load(BASE/'sampling_arrays.npz')[k])
    np.savez_compressed(RUN/'sampling_arrays.npz',**arrays)
    (RUN/'phase_manifest.json').write_text((DATA/'meta/fine_grip_phase_manifest.json').read_text())
    weights=arrays['weights'];eligible=arrays['eligible_train'];correction=arrays['episode']>=160
    for name in ['split_manifest.json','train_stats.json']:assert (RUN/name).read_bytes()==(BASE/name).read_bytes()
    audit=dict(scheme='fine_grip_with_recovery_ACT',stage_weights=dict(zip(NAMES,STAGE_WEIGHTS.tolist())),
        preclose_seconds=1.,recovery_weight=2,overlap_rule='maximum',loss_weight=1,
        original_mass=float(weights[~correction].sum()),correction_mass=float(weights[correction].sum()),
        phase_probability_mass={name:float(weights[phase==i].sum()) for i,name in enumerate(NAMES)},
        recovery_probability_mass=float(weights[recovery].sum()),
        training_windows=int(eligible.sum()),validation_windows=int(arrays['eligible_validation'].sum()),
        train_episodes=190,validation_episodes=21,chunk_size=16,
        first_close_frames_verified=len(starts),all_first_close_phase_weights=2.,
        first_close_probability_mass=sum(x['sampling_probability'] for x in starts),
        complete_approved_windows=True,heldout_excluded=True,split_preserved=True,stats_preserved=True,
        split_sha256=hashlib.sha256((RUN/'split_manifest.json').read_bytes()).hexdigest())
    write('sampling_audit.json',audit);write('first_close_verified.json',starts)
    # Verify the reused cache's completed metadata and a sample from all episodes.
    cache=Path('/home/lp531/smolvla_data/act_rgb224_float32_cache_20261015');assert (cache/'completed.json').exists()
    for r in split['source_records']:
        for cam in ['wrist','global','global_right']:
            v=np.load(cache/f'observation.images.{cam}'/f"episode_{r['episode_index']:06d}.npy",mmap_mode='r')
            assert v.shape==(r['frames'],3,224,224) and v.dtype==np.float32
            assert np.isfinite(v[0]).all()
    write('cache_reuse_verified.json',dict(cache=str(cache),episodes=211,files=633,rebuild=False,video_decoding_in_training=False,original_verification=str(BASE/'cache_verified.json')))
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
