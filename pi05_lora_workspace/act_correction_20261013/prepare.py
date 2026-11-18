"""Freeze an episode split, approved 16-action windows and train-only statistics."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sampling import build_weights
RUN=Path(__file__).resolve().parent
WORK=RUN.parent
DATA=Path('/home/lp531/smolvla_data/z1_bottle_with_fragments_20261012')

def write(name,value):
    target=RUN/name
    temp=target.with_suffix(target.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temp.replace(target)

def prepare():
    phase=json.loads((WORK/'fine_grip_recovery_base_20261012/phase_manifest.json').read_text())
    mapped=json.loads((WORK/'correction_windows_20261012/mapped_windows.json').read_text())
    rows=mapped['records']
    assert len(rows)==len(phase)==211
    # Keep duplicate raw recordings in the same split if any exist.
    groups={}
    for r in phase:groups.setdefault(r['source_frames_sha256'],[]).append(r)
    rng=np.random.default_rng(20261013)
    val=set()
    for source,count in [('original',16),('old_correction',3),('new_fragment',2)]:
        candidates=[g for g in groups.values() if g[0]['source']==source]
        for i in rng.permutation(len(candidates))[:count]:val.update(r['episode_index'] for r in candidates[i])
    train=set(range(211))-val
    assert not {r['source_frames_sha256'] for r in phase if r['episode_index'] in train}&{r['source_frames_sha256'] for r in phase if r['episode_index'] in val}
    table=pq.read_table(DATA/'data').sort_by([('index','ascending')])
    episode=np.asarray(table['episode_index']);frame=np.asarray(table['frame_index'])
    actions=np.asarray(table['action'].to_pylist(),dtype=np.float32)
    states=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
    assert len(episode)==47135 and set(np.unique(actions[:,-1]))=={0.,1.}
    assert np.isfinite(actions).all() and np.isfinite(states).all()
    eligible,weights=build_weights(episode,frame,rows,train,16)
    valid_val=np.zeros(len(episode),bool)
    for r in rows:
        ep=r['episode_index'];mask=episode==ep
        assert int(mask.sum())==r['frame_count']
        if ep in val:valid_val[mask]=(frame[mask]>=r['anchor_start'])&(frame[mask]+15<=r['anchor_end_inclusive'])
        assert (eligible[mask] if ep in train else valid_val[mask]).any(),ep
    np.savez_compressed(RUN/'sampling_arrays.npz',episode=episode,frame=frame,eligible_train=eligible,eligible_validation=valid_val,weights=weights)
    # Data statistics are computed from training episodes only; image stats are ImageNet constants.
    selected=np.isin(episode,list(train))
    stats={}
    for key,values in [('action',actions),('observation.state',states)]:
        x=values[selected].astype(np.float64)
        stats[key]=dict(mean=x.mean(0).tolist(),std=x.std(0).tolist(),min=x.min(0).tolist(),max=x.max(0).tolist(),count=[len(x)])
    for cam in ('wrist','global','global_right'):
        stats['observation.images.'+cam]=dict(mean=[[[v]] for v in [.485,.456,.406]],std=[[[v]] for v in [.229,.224,.225]])
    write('train_stats.json',stats)
    write('split_manifest.json',dict(seed=20261013,train_episodes=sorted(train),validation_episodes=sorted(val),
          validation_by_source=dict(Counter(r['source'] for r in phase if r['episode_index'] in val)),
          source_records=phase,source_hash_overlap=False))
    write('sampling_audit.json',dict(train_episodes=len(train),validation_episodes=len(val),
          training_windows=int(eligible.sum()),validation_windows=int(valid_val.sum()),
          original_mass=float(weights[episode<160].sum()),correction_mass=float(weights[episode>=160].sum()),
          fine_stage_weighting=False,full_window_inside_approved_interval=True,chunk_size=16,
          train_stats_frames=int(selected.sum()),source_dataset=str(DATA),
          mapped_windows_sha256=hashlib.sha256((WORK/'correction_windows_20261012/mapped_windows.json').read_bytes()).hexdigest()))
    print((RUN/'sampling_audit.json').read_text(),flush=True)
if __name__=='__main__':prepare()
