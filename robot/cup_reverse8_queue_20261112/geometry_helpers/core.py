"""Pure collection protocol; importing this module never starts ROS."""
import hashlib
import json
import os
import math
from pathlib import Path

QUOTAS = [('left',8),('right',8),('far',8),('near',4),('low',4),('high',4),
          ('pitch_tilt',8),('roll_tilt',4),('left_tilt',4),('right_tilt',4),('close_ready',4)]

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def make_tasks():
    tasks=[]
    for kind,count in QUOTAS:
        for i in range(count):
            scale=(.5,1.)[i%2]
            lateral=.006*scale;depth=.010*scale;vertical=.002*scale
            offset=[0.,0.,0.];tilt=[0.,0.,0.]
            if kind in ('left','left_tilt'):offset[1]=lateral
            if kind in ('right','right_tilt'):offset[1]=-lateral
            if kind=='far':offset[0]=-depth
            if kind=='near':offset[0]=.003*scale
            if kind=='low':offset[2]=-vertical
            if kind=='high':offset[2]=vertical
            if 'tilt' in kind:
                angle=(1.5,3.)[i%2]*(1 if (i//2)%2==0 else -1)
                tilt[1 if kind=='pitch_tilt' else 0]=angle
            slot=(len(tasks)*7)%20;grid=[slot%5,slot//5]
            split='validation' if tuple(grid) in ((1,1),(2,3)) else 'train'
            tasks.append(dict(id=f'cup_{kind}_{i+1:03d}',kind=kind,grid=grid,split=split,
                              offset_approach_m=offset,rotation_tool_deg=tilt))
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

