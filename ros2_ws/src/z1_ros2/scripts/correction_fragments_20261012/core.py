"""Pure collection protocol; importing this module never starts ROS."""
import hashlib
import json
import os
import math
from pathlib import Path

QUOTAS = [('left', 4), ('low_tilt', 2), ('right', 1), ('left_tilt', 1), ('low', 3), ('right_low', 1), ('tilt', 1), ('far', 2), ('far_tilt', 2), ('right_tilt', 1), ('close_ready', 2)]

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def make_tasks(amplitudes):
    result=[]
    for kind,n in QUOTAS:
        for i in range(n):
            scale=[.65,1.0][i%2];x,y,z=[float(v)*scale for v in amplitudes['xyz_m']]
            delta={'low':[0,0,-z],'high':[0,0,z],'left':[0,y,0],'right':[0,-y,0],
                   'far':[-x,0,0],'near':[x,0,0],'tilt':[0,0,0],'low_tilt':[0,0,-z]}[kind]
            tilt=[0.,0.,0.]
            if 'tilt' in kind:
                tilt[(i//2)%2]=amplitudes['tilt_deg']*scale*(1 if (i//4)%2==0 else -1)
            cell=(i*7)%20
            result.append(dict(id=f'{kind}_{i+1:03d}',kind=kind,grid=[cell%5,cell//5],
                               offset_approach_m=delta,rotation_tool_deg=tilt))
    return result

def completed_ids(root, plan_hash):
    result=set()
    for path in Path(root).glob('episode_[0-9][0-9][0-9][0-9][0-9][0-9]'):
        try:
            info=json.loads((path/'correction.json').read_text())
            summary=json.loads((path/'episode_summary.json').read_text())
            if (info.get('accepted') is True and info.get('plan_hash')==plan_hash
                    and summary.get('fragment_success') is True and summary.get('physical_grasp_ok') is True and (path/'frames.jsonl').stat().st_size>0):
                result.add(info['task_id'])
        except (OSError,ValueError,KeyError):
            continue
    return result

def strict_final(status):
    p=status.get('object_position_m',[])
    return bool(status.get('task_success') and status.get('release_sequence_complete')
                and len(p)==3 and abs(p[0]-.32)<=.0278 and abs(p[1]-.14)<=.0278 and 0<=p[2]<=.025)

def alignment_step(actual, desired):
    error=[float(a)-float(b) for a,b in zip(actual,desired)]
    if len(error)!=3 or not all(math.isfinite(x) for x in error):raise ValueError('invalid local position')
    norm=math.sqrt(sum(x*x for x in error))
    if norm>.025:raise RuntimeError('到位残差超过25mm，拒绝当作小幅精对齐')
    scale=min(1.,.006/max(norm,1e-9))
    return [x*scale for x in error]

def recovery_waypoints(position, orientation, reference_p, reference_q, kind):
    p=list(position);q=list(orientation);r=list(reference_p);rq=list(reference_q)
    # Lift without rotating near the table; rotate only in the raised corridor.
    raised=[p[0],p[1],max(p[2],r[2])+.045]
    return [dict(stage='lift_clear',position=raised,orientation=q),
            dict(stage='correct_orientation',position=raised,orientation=rq),
            dict(stage='align_above',position=[r[0],r[1],raised[2]],orientation=rq),
            dict(stage='approach',position=r,orientation=rq)]
