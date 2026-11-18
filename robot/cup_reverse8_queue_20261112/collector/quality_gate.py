"""Shared physical-recording retake gate; inputs are read, never rewritten."""
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0,'/home/lp531/robot/dataset_trajectory_review')
import review
import screen_fruit_motion as motion
from screen_cup_training import cup_command_labels

_CHAIN=None
_FIXED=None


def screen(folder,maximum_wait_s=8.):
    global _CHAIN,_FIXED
    root=Path(folder)
    if _CHAIN is None:
        _CHAIN=review.chain_from_urdf(review.DEFAULT_URDF)
        robot=ET.parse(review.DEFAULT_URDF).getroot()
        _FIXED=np.fromstring(robot.find("joint[@name='gripperStatorLink']/origin").get('xyz'),sep=' ')
    rows=[json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
    q=np.asarray([r['observation.state'] for r in rows],float)
    assert q.shape==(len(rows),7) and np.isfinite(q).all()
    times=np.asarray([r['simulation_time'] for r in rows],float)
    assert len(rows)>2 and np.allclose(np.diff(times),.1,atol=1e-5,rtol=0)
    position,rotation=review.forward(q[:,:6],_CHAIN)
    points=position+np.einsum('nij,j->ni',rotation,_FIXED)
    meta=review.read_json(root/'correction.json',{})
    expert=review.event_mask(meta,times)
    mask=review.read_json(root/'training_mask.json',{}).get('expert_frame_mask')
    if mask is not None:
        if len(mask)!=len(rows):raise ValueError('Expert mask/frame count differs')
        expert &= np.asarray(mask,bool)
    labels=review.read_json(root/'gripper_labels.json',{}).get('labels')
    if labels is not None:
        if len(labels)!=len(rows) or not np.isin(labels,[0,1]).all():raise ValueError('Jaw label/frame mismatch')
        labels=np.asarray(labels,int)
    elif rows[0]['target_object']=='apple':labels=motion.command_labels(rows,root)[0]
    else:labels=cup_command_labels(rows)
    phases=np.asarray([r['phase'] for r in rows],dtype=object)
    for stages,start,end in [(meta.get('stages',[]),'start_sim_time','end_sim_time'),
                             (meta.get('recovery_stages',[]),'start','end')]:
        for stage in stages:
            if not all(k in stage for k in [start,end,'stage']):continue
            select=(times>=stage[start])&(times<stage[end])
            phases[select]=[str(p)+'#'+stage['stage'] for p in phases[select]]
    segments=motion.reverse_segments(points,phases,labels,expert,times,10.,
        bool(meta.get('fragment_goal')),motion.Rules())
    maximum=max([s['max_reverse_excursion_m'] for s in segments],default=0.)
    waits=review.analyze(root/'frames.jsonl',_CHAIN,review.Settings())
    intervals=[f for f in waits['findings'] if f['type'] in ['长停顿候选','末尾停留偏长']]
    longest=max([f['recorded_s'] for f in intervals],default=0.)
    report=dict(passed=maximum<.08 and longest<maximum_wait_s,
        max_reverse_excursion_m=maximum,reverse_limit_m=.08,
        longest_candidate_wait_s=longest,maximum_allowed_wait_s=maximum_wait_s,
        frames_sha256=waits['frames_sha256'],motion_segments=segments,wait_intervals=intervals,
        deliberate_interventions_excluded=True)
    review.atomic_json(root/'retake_quality.json',report)
    # Keep compatibility with the independent correction collector's audit.
    review.atomic_json(root/'time_screen.json',dict(passed=longest<maximum_wait_s,
        longest_candidate_wait_s=longest,maximum_allowed_wait_s=maximum_wait_s,
        frames_sha256=waits['frames_sha256'],intervals=intervals))
    if not report['passed']:
        raise ValueError(f'Retake rejected: reverse={maximum*100:.2f}cm (limit<8cm); wait={longest:.2f}s (limit<8s)')
    return report
