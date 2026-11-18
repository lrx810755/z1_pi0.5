"""Audit downloaded YAM examples; do not treat five episodes as the full corpus."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
rows=[]
for f in sorted(ROOT.glob('yam_*.npz')):
    z=np.load(f,allow_pickle=False);a=z['leader_actions'][:,:7];q=z['follower_joints'][:,:7];t=z['timestamp']
    assert len(t)==len(a)==len(q) and np.isfinite(a).all() and np.isfinite(q).all()
    # Model card: native convention 0 closed, 1 open. Inspect first downward
    # crossing of .3 after reaching .5; this is a proxy, not physical contact.
    opened=np.flatnonzero(a[:,-1]>=.5)
    crossing=np.flatnonzero((a[1:,-1]<.3)&(a[:-1,-1]>=.3))+1
    crossing=crossing[crossing>opened[0]] if len(opened) else []
    c=int(crossing[0]) if len(crossing) else None
    hold=None
    if c is not None:
        b=c
        while b>0 and np.max(np.abs(q[b-1,:6]-q[c,:6]))<=.01:b-=1
        hold=float(t[c]-t[b])
    error=np.max(np.abs(a[:,:6]-q[:,:6]),axis=1)
    rows.append({'file':f.name,'frames':len(t),'duration_s':float(t[-1]-t[0]),
        'step_dt_quantiles_s':np.quantile(np.diff(t),[0,.5,.95,1]).tolist(),
        'arm_tracking_error_quantiles_rad':np.quantile(error,[.5,.95,1]).tolist(),
        'first_gripper_downcross_03':c,'arm_stationary_before_cross_s':hold,
        'joint_angles_finite':True,'timestamps_strictly_increasing':bool((np.diff(t)>0).all())})
force=[json.loads(s) for s in (ROOT/'forcesight_episodes.jsonl').read_text().splitlines()]
report={'yam_sample_scope':'Episodes 1,21,41,61,81 of 100, raw NPZ only. No model evaluation or camera synchronization revalidation.',
        'yam':rows,'forcesight_scope':'All episode metadata, task table and pi05_6_3 info only; no full image/action parquet downloaded.',
        'forcesight_tasks_counts':dict(Counter(task for r in force for task in r['tasks'])),
        'forcesight_episode_frames':{'min':min(r['length'] for r in force),'median':float(np.median([r['length'] for r in force])), 'max':max(r['length'] for r in force)}}
(ROOT/'public_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
