"""Read-only screening of saved demonstrations; no training masks are changed."""
import csv
import datetime
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

DATA = Path('/home/lp531/smolvla_data/bottle_correction60_20261111')
OUT = Path('/home/lp531/z1_queue_logs/trajectory_time_audit_20261111')
URDF = Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf')


def runs(mask):
    edge = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def fk(q):
    robot = ET.parse(URDF).getroot()
    count = len(q)
    transforms = np.repeat(np.eye(4)[None], count, axis=0)
    for i in range(6):
        joint = robot.find(f"joint[@name='joint{i+1}']")
        origin = joint.find('origin')
        fixed = np.eye(4)
        fixed[:3, 3] = np.fromstring(origin.get('xyz'), sep=' ')
        fixed[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy'), sep=' ')).as_matrix()
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        moving = np.repeat(np.eye(4)[None], count, axis=0)
        moving[:, :3, :3] = Rotation.from_rotvec(q[:, i, None] * axis).as_matrix()
        transforms = transforms @ fixed @ moving
    return transforms[:, :3, 3], transforms[:, :3, :3]


def low_motion_mask(q, positions, rotations, gripper=None, command=None):
    """Require bounded displacement throughout a one-second window, not one frame."""
    count = len(q)
    width = 11  # ten intervals at10Hz
    mask = np.zeros(count, bool)
    for start in range(count - width + 1):
        end = start + width
        if np.max(np.ptp(q[start:end], axis=0)) > .006:
            continue
        if np.linalg.norm(np.ptp(positions[start:end], axis=0)) > .002:
            continue
        rel = rotations[start].T @ rotations[start:end]
        angle = Rotation.from_matrix(rel).magnitude()
        if np.max(angle) > np.deg2rad(1.):
            continue
        if gripper is not None and np.ptp(gripper[start:end]) > .012:
            continue
        if command is not None and np.max(np.ptp(command[start:end, :6], axis=0)) > .006:
            continue
        if command is not None and gripper is not None and np.ptp(command[start:end, 6]) > .015:
            continue
        mask[start:end] = True
    return mask


def read_physics(path, times):
    stamps, positions, holding, heights = [], [], [], []
    with (path / 'physics_status.jsonl').open() as stream:
        for line in stream:
            d = json.loads(line)
            status = d['status']
            position = status.get('object_position_m')
            if position is None or len(position) != 3:
                continue
            stamps.append(float(status.get('object_pose_stamp_s', d['simulation_time'])))
            positions.append(position)
            holding.append(status.get('bilateral_contact', False))
    stamps, positions, holding = np.asarray(stamps), np.asarray(positions), np.asarray(holding)
    order = np.argsort(stamps, kind='stable')
    stamps, positions, holding = stamps[order], positions[order], holding[order]
    right = np.searchsorted(stamps, times).clip(0, len(stamps) - 1)
    left = np.maximum(right - 1, 0)
    ids = np.where(np.abs(stamps[left] - times) < np.abs(stamps[right] - times), left, right)
    return positions[ids], holding[ids], np.abs(stamps[ids] - times)


def phase_statistics(phases, fps):
    return {k: round(v / fps, 2) for k, v in Counter(phases).items()}


def episode(path):
    summary = json.loads((path / 'episode_summary.json').read_text())
    meta = json.loads((path / 'correction.json').read_text())
    rows = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
    masks = json.loads((path / 'training_mask.json').read_text())
    q = np.asarray([r['observation.state'] for r in rows], float)
    command = np.asarray([r['action'] for r in rows], float)
    times = np.asarray([r['simulation_time'] for r in rows], float)
    t = times - times[0]
    fps = float(summary['fps'])
    assert q.shape == command.shape == (len(rows), 7)
    assert np.isfinite(q).all() and np.isfinite(command).all()
    assert np.allclose(np.diff(times), 1 / fps, atol=1e-6)
    expert = np.asarray(masks['expert_frame_mask'], bool)
    ee, rotations = fk(q[:, :6])
    object_position, holding, lag = read_physics(path, times)
    arm_static = low_motion_mask(q[:, :6], ee, rotations)
    robot_static = low_motion_mask(q[:, :6], ee, rotations, q[:, 6], command)
    phases = np.asarray([r['phase'] for r in rows])
    close, release = int(meta['close_command_frame']), int(meta['open_command_frame'])
    assert 0 <= close < release < len(rows)
    withdrawal = [e['time'] - times[0] for e in meta['events'] if e['event'] == 'withdrawal_measured']
    withdrawal_end = withdrawal[-1] if withdrawal else float(t[-1])
    release_tail = t >= withdrawal_end - 1e-7
    initial = t <= .35
    eligible = expert & ~release_tail & ~initial
    candidates = []
    for a, b in runs(robot_static & eligible):
        duration = (b - a) / fps
        if duration < 3.:
            continue
        span = float(np.linalg.norm(np.ptp(ee[a:b], axis=0)))
        # A long, slowly translating path is reported separately from a hold.
        slow = span > .004 or np.max(np.ptp(q[a:b, :6], axis=0)) > .012
        replan_events = [e for e in meta['events'] if e['event'] == 'stale_leg_replanned'
                         and t[a] - .3 <= e['time'] - times[0] <= t[b-1] + .7]
        candidates.append(dict(episode=path.name, kind=meta['task']['kind'],
            start_s=round(float(t[a]), 2), end_s=round(float(t[b-1] + 1/fps), 2),
            duration_s=round(duration, 2), start_frame=int(a), end_frame_exclusive=int(b),
            phase_counts=dict(Counter(phases[a:b])),
            ee_range_mm=round(span*1000, 3),
            joint_range_deg=round(float(np.max(np.ptp(q[a:b, :6], axis=0)))*180/math.pi, 3),
            gripper_range_rad=round(float(np.ptp(q[a:b, 6])), 5),
            object_range_mm=round(float(np.linalg.norm(np.ptp(object_position[a:b], axis=0)))*1000, 3),
            replan_event_count=len(replan_events), category='slow_drift' if slow else 'near_stationary',
            priority='high' if duration >= 5 and not slow else 'review',
            interpretation='Candidate only; may include necessary contact validation or planning.',
            path=str(path)))
    first_box = np.flatnonzero((np.linalg.norm(object_position[:, :2] - [.32, .14], axis=1) <= .035)
                              & (object_position[:, 2] >= .14) & holding & (np.arange(len(rows)) >= close))
    box = int(first_box[0]) if len(first_box) else None
    load = np.flatnonzero((object_position[:, 2] >= object_position[0, 2] + .03)
                          & holding & (np.arange(len(rows)) >= close))
    first_lift = int(load[0]) if len(load) else None
    end = len(rows) / fps
    metrics = dict(episode=path.name, kind=meta['task']['kind'], path=str(path),
        grid=meta['task']['grid'], frames=len(rows), duration_s=round(end, 2),
        expert_s=round(np.sum(expert)/fps, 2), intervention_s=round(np.sum(~expert)/fps, 2),
        arm_static_s=round(np.sum(arm_static)/fps, 2),
        robot_static_s=round(np.sum(robot_static)/fps, 2),
        expert_static_s=round(np.sum(robot_static & expert)/fps, 2),
        candidate_hold_s=round(sum(c['duration_s'] for c in candidates if c['category']=='near_stationary'), 2),
        longest_candidate_s=round(max((c['duration_s'] for c in candidates if c['category']=='near_stationary'), default=0.), 2),
        long_hold_count=sum(c['category']=='near_stationary' for c in candidates),
        severe_hold_count=sum(c['priority']=='high' for c in candidates),
        final_close_s=round(float(t[close]), 2), release_command_s=round(float(t[release]), 2),
        close_to_release_s=round(float(t[release]-t[close]), 2),
        first_lift_s=None if first_lift is None else round(float(t[first_lift]), 2),
        first_box_entry_s=None if box is None else round(float(t[box]), 2),
        close_to_box_s=None if box is None else round(float(t[box]-t[close]), 2),
        box_to_release_s=None if box is None else round(float(t[release]-t[box]), 2),
        post_withdrawal_tail_s=round(max(0., end-withdrawal_end), 2),
        phase_durations_s=phase_statistics(phases, fps),
        replan_count=sum(e['event']=='stale_leg_replanned' for e in meta['events']),
        physics_alignment_max_s=round(float(lag.max()), 6),
        frames_sha256=hashlib.sha256((path/'frames.jsonl').read_bytes()).hexdigest())
    return metrics, candidates, dict(t=t, q=q, ee=ee, command=command, object=object_position,
        expert=expert, robot_static=robot_static, eligible=eligible, phases=phases, meta=meta)


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        fields = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})


def plots(metrics, intervals, signals):
    kinds = list(dict.fromkeys(m['kind'] for m in metrics))
    fig, axes = plt.subplots(1,2,figsize=(13,4),layout='constrained')
    axes[0].boxplot([[m['duration_s'] for m in metrics if m['kind']==k] for k in kinds],
                   labels=kinds, showfliers=True)
    axes[0].tick_params(axis='x', labelrotation=25)
    axes[0].set_ylabel('Simulation seconds')
    axes[0].set_title('Total duration by correction category')
    for kind in kinds:
        group=[m for m in metrics if m['kind']==kind]
        axes[1].scatter([m['duration_s'] for m in group], [m['candidate_hold_s'] for m in group], label=kind)
    axes[1].set_xlabel('Total duration (s)');axes[1].set_ylabel('Candidate stationary time >=3s (s)')
    axes[1].legend(fontsize=8);axes[1].set_title('Long recording does not always mean long idle')
    fig.savefig(OUT/'duration_overview.png',dpi=160);plt.close(fig)
    ranked=sorted(metrics,key=lambda m:(m['candidate_hold_s'],m['longest_candidate_s']),reverse=True)[:6]
    fig,axes=plt.subplots(len(ranked),1,figsize=(13,2.3*len(ranked)),layout='constrained')
    for ax,m in zip(np.atleast_1d(axes),ranked):
        s=signals[m['episode']];t=s['t']
        q=median_filter(s['q'][:,:6],size=(3,1),mode='nearest')
        arm_speed=np.max(np.abs(np.diff(q,axis=0)),axis=1)*10
        ax.plot(t[1:],arm_speed,label='Max arm joint speed (rad/s)',lw=.8)
        ax.plot(t[1:],np.abs(np.diff(s['q'][:,6]))*10,label='Gripper speed (rad/s)',lw=.65,alpha=.7)
        for a,b in runs(~s['expert']):ax.axvspan(t[a],t[b-1]+.1,color='gray',alpha=.3)
        for interval in intervals:
            if interval['episode']==m['episode']:
                ax.axvspan(interval['start_s'],interval['end_s'],color='orange',alpha=.4)
        ax.axvline(m['final_close_s'],color='green',ls='--',lw=1)
        ax.axvline(m['release_command_s'],color='purple',ls='--',lw=1)
        ax.set_title(f"{m['episode']} | {m['kind']} | candidate holds={m['candidate_hold_s']:.1f}s",fontsize=10)
        ax.set_ylabel('rad/s');ax.set_ylim(bottom=0);ax.set_xlabel('Episode simulation time (s)')
    np.atleast_1d(axes)[0].legend(fontsize=8,loc='upper right')
    fig.suptitle('Orange: candidate low-progress holds; gray: excluded deliberate interventions',fontsize=11)
    fig.savefig(OUT/'long_hold_candidates.png',dpi=150);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    snapshot=sorted(p.parent for p in DATA.glob('episode_*/episode_summary.json'))
    started=datetime.datetime.now().astimezone().isoformat()
    metrics, intervals, signals=[],[],{}
    for path in snapshot:
        m,c,s=episode(path);metrics.append(m);intervals.extend(c);signals[path.name]=s
    by_kind={}
    for kind in sorted({m['kind'] for m in metrics}):
        group=[m for m in metrics if m['kind']==kind]
        duration=np.asarray([m['duration_s'] for m in group])
        adjusted=np.asarray([m['expert_s'] for m in group])
        median=float(np.median(adjusted));mad=float(np.median(np.abs(adjusted-median)))
        limit=max(median*1.5,median+3*1.4826*mad)
        for m in group:m['adjusted_duration_outlier']=bool(m['expert_s']>limit)
        by_kind[kind]=dict(count=len(group),median_total_s=round(float(np.median(duration)),2),
            range_total_s=[round(float(duration.min()),2),round(float(duration.max()),2)],
            median_expert_s=round(median,2),expert_duration_screen_threshold_s=round(limit,2),
            episodes_with_candidate_holds=sum(m['long_hold_count']>0 for m in group),
            candidate_hold_s=round(sum(m['candidate_hold_s'] for m in group),2))
    result=dict(snapshot_at=started,data_root=str(DATA),episode_count=len(metrics),snapshot_episodes=[p.name for p in snapshot],
        scope='Saved bottle supplemental corrections only; collection continues independently.',
        criteria=dict(window_s=1.,max_ee_range_mm=2.,max_orientation_range_deg=1.,
            max_joint_range_rad=.006,max_gripper_range_rad=.012,min_candidate_duration_s=3.,
            high_priority_duration_s=5.,exclude_deliberate_interventions=True,
            exclude_initial_s=.35,exclude_after_withdrawal=True,
            note='Screening heuristics, not automatic rejection or confirmed unnecessary waiting.'),
        by_kind=by_kind,
        totals=dict(duration_s=round(sum(m['duration_s'] for m in metrics),2),
            intervention_s=round(sum(m['intervention_s'] for m in metrics),2),
            expert_s=round(sum(m['expert_s'] for m in metrics),2),
            expert_static_s=round(sum(m['expert_static_s'] for m in metrics),2),
            candidate_hold_s=round(sum(m['candidate_hold_s'] for m in metrics),2),
            candidate_intervals=len(intervals),
            episodes_with_candidates=sum(m['long_hold_count']>0 for m in metrics),
            episodes_with_high_priority=sum(m['severe_hold_count']>0 for m in metrics)),
        episodes=metrics,intervals=sorted(intervals,key=lambda c:c['duration_s'],reverse=True))
    (OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    csv_write(OUT/'episode_metrics.csv',metrics)
    csv_write(OUT/'candidate_intervals.csv',result['intervals'])
    plots(metrics,intervals,signals)
    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','intervals','snapshot_episodes')},ensure_ascii=False,indent=2))
    print('TOP_INTERVALS',json.dumps(result['intervals'][:12],ensure_ascii=False))


if __name__=='__main__':
    main()
