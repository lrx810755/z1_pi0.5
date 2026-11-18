"""No-motion selection of complete preparation and recovery routes."""
import numpy as np
from scipy.spatial.transform import Rotation

def routes(position, orientation, object_position, task):
    p=np.array(position,dtype=float);q=list(orientation)
    # The cup uses a shallow side pinch. Define horizontal errors from the
    # actual approved tool approach, including azimuth alternatives.
    forward=Rotation.from_quat(q).apply([1.,0.,0.])[:2]
    norm=np.linalg.norm(forward)
    if norm<1e-6:raise ValueError('undefined horizontal cup approach direction')
    forward/=norm;left=np.array([-forward[1],forward[0]])
    dx,dy,dz=task['offset_approach_m']
    bad_q=(Rotation.from_quat(q)*Rotation.from_euler('xyz',task['rotation_tool_deg'],degrees=True)).as_quat().tolist()
    # Only the clearance corridor varies. Never dilute the requested error or
    # silently replace an unreachable tilt/low example with a baseline example.
    retreat=float(task.get('preparation_retreat_m',.004))
    choices=[(0.,.012)] if task['kind']=='close_ready' else [(retreat,.018),(retreat,.025)]
    for retreat,lift in choices:
        anchor=p.copy();anchor[:2]-=retreat*forward
        bad=anchor.copy();bad[:2]+=dx*forward+dy*left;bad[2]+=dz
        high_z=max(p[2],bad[2])+lift
        raised=[bad[0],bad[1],high_z]
        preparation=[dict(stage='prepare_clearance',position=raised,orientation=q,linear=False),
                     dict(stage='prepare_tilt',position=raised,orientation=bad_q,linear=False),
                     dict(stage='prepare_descend',position=bad.tolist(),orientation=bad_q,linear=True)]
        recovery=[dict(stage='lift_clear',position=raised,orientation=bad_q,linear=True),
                  dict(stage='correct_orientation',position=raised,orientation=q,linear=False),
                  dict(stage='align_above',position=[p[0],p[1],high_z],orientation=q,linear=True),
                  dict(stage='approach',position=p.tolist(),orientation=q,linear=True)]
        if task['kind']=='close_ready':
            bad=p.copy();bad_q=q;preparation[-1].update(position=p.tolist(),orientation=q);recovery=[]
        yield dict(anchor_position=anchor.tolist(),bad_position=bad.tolist(),bad_orientation=bad_q,
                   retreat_m=retreat,lift_m=lift,preparation=preparation,recovery=recovery)

def choose_route(candidates, plan_route):
    failures=[]
    for candidate in candidates:
        try:return candidate,plan_route(candidate),failures
        except RuntimeError as exc:failures.append(str(exc))
    raise RuntimeError('纠正全路径预检未通过: '+' | '.join(failures[-3:]))

def trajectory_points(trajectory, initial):
    points=[list(initial)]
    for point in trajectory.points:
        q=list(initial)
        for name,value in zip(trajectory.joint_names,point.positions):
            if name in ['joint1','joint2','joint3','joint4','joint5','joint6']:q[int(name[-1])-1]=float(value)
            elif name=='jointGripper':q[6]=float(value)
        points.append(q)
    return points
