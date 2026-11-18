"""Pure collection protocol; importing this module never starts ROS."""
import hashlib
import json
import os
import math
from pathlib import Path

QUOTAS = [('approach_recovery',36),('post_shift_realign',12)]

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def make_tasks():
    evidence=json.loads(Path(__file__).with_name('failure_directions.json').read_text())
    tasks=[]
    for rep,scale in enumerate((.55,.8,1.)):
        for e in evidence:
            slot=(len(tasks)*7)%20;grid=[slot%5,slot//5]
            tasks.append(dict(id=f'cup_failure_{e["eval_episode"]:02d}_level{rep+1}',kind='approach_recovery',grid=grid,
                split='validation' if tuple(grid) in ((1,1),(2,3)) else 'train',
                offset_approach_m=[round(v*scale,6) for v in e['bounded_offset_m']],
                rotation_tool_deg=[round(v*scale,3) for v in e['relative_tool_rotation_deg']],
                preparation_retreat_m=.012,freeze_policy='pre_approach_only',failure_evidence=e,
                perturbation_scale=scale,source_note='Failure-derived bounded perturbation, not an exact replay. Random fresh scene.'))
    for i in range(12):
        slot=(len(tasks)*7)%20;grid=[slot%5,slot//5];m=(.008,.012,.015)[i//4]
        shift=([m,0,0],[-m,0,0],[0,m,0],[0,-m,0])[i%4]
        tasks.append(dict(id=f'cup_post_shift_{i+1:02d}',kind='post_shift_realign',grid=grid,
            split='validation' if tuple(grid) in ((1,1),(2,3)) else 'train',offset_approach_m=[0,0,0],
            rotation_tool_deg=[0,0,0],requested_shift_world_m=shift,preparation_retreat_m=.012,
            freeze_policy='pre_approach_only',source_note='Cup relocated during scene preparation; record recovery from stale target pose, never teleport during recording.'))
    return tasks

def completed_ids(root, plan_hash):
    result=set()
    accepted_hashes={plan_hash}
    try:
        plan=json.loads((Path(root)/'correction_plan.json').read_text())
        if plan.get('hash')==plan_hash:accepted_hashes.update(plan.get('accepted_plan_hashes',[]))
    except (OSError,ValueError):pass
    for path in Path(root).glob('episode_[0-9][0-9][0-9][0-9][0-9][0-9]'):
        try:
            info=json.loads((path/'correction.json').read_text())
            summary=json.loads((path/'episode_summary.json').read_text())
            if (info.get('accepted') is True and info.get('plan_hash') in accepted_hashes
                    and summary.get('target_object')=='cup' and summary.get('fragment_success') is True and summary.get('physical_grasp_ok') is True and (path/'frames.jsonl').stat().st_size>0):
                result.add(info['task_id'])
        except (OSError,ValueError,KeyError):
            continue
    return result

