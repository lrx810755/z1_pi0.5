"""Read-only audit of all raw episodes and the exact training parquet values."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RAW = Path('/home/lp531/smolvla_data/z1_tabletop_bottle_recovery_boundary_only_prepared_v1')
TRAIN = Path('/home/lp531/smolvla_data/z1_tabletop_bottle_pi05_lora_v1')
manifest = json.loads(Path('/home/lp531/pi05_lora_workspace/split_manifest.json').read_text())
table = pq.read_table(TRAIN / 'data').to_pydict()
ids = np.asarray(table['episode_index'])
aggregate = defaultdict(list)
episodes = []
phases = Counter()
def quant(values):
    return dict(zip(['min','p50','p95','p99','max'], np.quantile(values,[0,.5,.95,.99,1]).tolist()))

for entry in manifest['episodes']:
    old = entry['original_episode_index']
    folder = RAW / f'episode_{old:06d}'
    rows = [json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
    state = np.asarray([r['observation.state'] for r in rows])
    action = np.asarray([r['action'] for r in rows])
    ts = np.asarray([r['timestamp'] for r in rows])
    actual = np.flatnonzero(ids == entry['episode_index'])
    parity = all(np.array_equal(np.asarray(table[key])[actual], arr.astype(np.float32)) for key,arr in [('observation.state',state),('action',action),('timestamp',ts)])
    assert parity, ('parquet changed source values',old)
    assert len(rows)==entry['frames'] and np.isfinite(state).all() and np.isfinite(action).all()
    assert np.allclose(np.diff(ts),.1) and [r['frame_index'] for r in rows]==list(range(len(rows)))
    c = int(np.flatnonzero(action[:,-1]>=-.75)[0])
    onset = int(np.flatnonzero(action[:,-1]>-.85)[0])
    begin=c
    while begin>0 and np.max(np.abs(state[begin-1,:6]-state[c,:6]))<=.01:
        begin-=1
    err = np.abs(state-action)
    step = np.max(np.abs(np.diff(action[:,:6],axis=0)),axis=1)
    next_target_error = np.max(np.abs(action[1:,:6]-state[:-1,:6]),axis=1)
    ages = [r['action_age_s'] for r in rows]
    camerr = [max(abs(r['wrist_image_stamp']-r['simulation_time']),abs(r['global_image_stamp']-r['simulation_time'])) for r in rows]
    pairerr = [abs(r['wrist_image_stamp']-r['global_image_stamp']) for r in rows]
    gaps = np.diff([r['simulation_time'] for r in rows])
    state_onset=np.flatnonzero(state[:,-1]>-.85)
    phases.update(r['phase'] for r in rows)
    item = dict(entry, first_close=c, close_onset=onset, stationary_before_close_s=(c-begin)/10,
                stationary_frames=c-begin, open_stationary_frames=int(sum(action[i,-1]<=-.85 for i in range(begin,c))),
                close_onset_state_rad=float(state[onset,-1]), close_threshold_state_rad=float(state[c,-1]),
                state_already_over_close_threshold=bool(state[c,-1]>=-.75),
                state_close_onset=int(state_onset[0]) if len(state_onset) else None,
                max_arm_action_jump_rad=float(step.max()), arm_jumps_gt_01=int((step>.1).sum()),
                next_target_vs_previous_measured_gt_01=int((next_target_error>.1).sum()),
                max_arm_tracking_error_rad=float(err[:,:6].max()),
                max_camera_sync_error_s=max(camerr), max_action_age_s=max(ages),
                raw_source=str(folder), task_success=json.loads((folder/'episode_summary.json').read_text())['task_success'])
    for k,v in [('arm_error',err[:,:6].max(axis=1)),('jaw_error',err[:,-1]),('arm_action_step',step),('next_target_vs_previous_measured',next_target_error),('action_age',ages),('camera_sync_error',camerr),('camera_pair_error',pairerr),('simulation_gap',gaps)]:
        aggregate[k].extend(v)
    episodes.append(item)

allstates=np.asarray(table['observation.state']);allactions=np.asarray(table['action'])
trainmask=ids<221
stats=json.loads((TRAIN/'meta/stats.json').read_text())
stats_match={k:all(np.allclose(np.quantile(a[trainmask],q,axis=0),stats[k][label],atol=1e-7) for q,label in [(.01,'q01'),(.99,'q99')]) for k,a in [('observation.state',allstates),('action',allactions)]}
report={'scope':'All 246 raw episodes, all 65773 training parquet rows; images checked separately.',
    'episodes':episodes,'summary':{'episodes':len(episodes),'frames':len(ids),'exact_raw_parquet_parity':True,
        'timestamp_grid_contiguous':True,'finite_state_actions':True,'train_only_quantiles_match':stats_match,
        'split_groups':{s:dict(Counter(e['group'] for e in episodes if e['split']==s)) for s in ['train','validation']},
        'phase_counts':dict(phases),'stationary_s':quant([e['stationary_before_close_s'] for e in episodes]),
        'stationary_total_frames':sum(e['stationary_frames'] for e in episodes),
        'open_stationary_total_frames':sum(e['open_stationary_frames'] for e in episodes),
        'close_onset_state':quant([e['close_onset_state_rad'] for e in episodes]),
        'close_threshold_state':quant([e['close_threshold_state_rad'] for e in episodes]),
        'state_already_closed_at_command_threshold':sum(e['state_already_over_close_threshold'] for e in episodes),
        'arm_jumps_gt_01':sum(e['arm_jumps_gt_01'] for e in episodes),
        'next_target_vs_previous_measured_gt_01':sum(e['next_target_vs_previous_measured_gt_01'] for e in episodes),
        'episodes_arm_jumps_gt_01':sum(e['arm_jumps_gt_01']>0 for e in episodes),
        'state_jaw_range':[float(allstates[:,-1].min()),float(allstates[:,-1].max())],
        'action_jaw_range':[float(allactions[:,-1].min()),float(allactions[:,-1].max())],
        'metrics':{k:quant(v) for k,v in aggregate.items()}}}
(ROOT/'local_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report['summary'],ensure_ascii=False,indent=2))
