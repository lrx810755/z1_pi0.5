"""Fragment-specific wait reductions; the parent retains final grasp validation."""
import copy
import math
import time
from preflight import trajectory_points


def angle(a,b):
    norm=math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    if not math.isfinite(norm) or norm<=0:return math.inf
    return 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b)))/norm))


def compact_route(route):
    result=dict(route)
    for phase in ('preparation','recovery'):
        kept=[]
        for step in route[phase]:
            if (kept and math.dist(step['position'],kept[-1]['position'])<1e-8
                    and angle(step['orientation'],kept[-1]['orientation'])<1e-8):
                continue
            kept.append(step)
        result[phase]=kept
    return result


def snapshot(node):
    with node._lock:
        return copy.deepcopy((node._object_poses,node._object_orientations))


def unchanged(node,scene):
    current,rotations=snapshot(node)
    poses,old_rotations=scene
    if not poses or set(poses)!=set(current):return False
    for name,(_,position) in poses.items():
        now,actual=current[name];delta=math.dist(position,actual)
        if not 0<=time.monotonic()-now<=getattr(node,'object_timeout',2.):return False
        if not math.isfinite(delta) or delta>.003:return False
        if name not in rotations or name not in old_rotations:return False
        if not 0<=time.monotonic()-rotations[name][0]<=getattr(node,'object_timeout',2.):return False
        if angle(rotations[name][1],old_rotations[name][1])>.035:return False
    return True


def plan(node,p,q,object_position,state=None,open_gripper=False):
    node._raise_if_stop_requested()
    target=(float(p[0]),float(p[1]),float(p[2])+max(0.,.06-float(object_position[2])))
    kwargs=dict(linear=True,position_tolerance=.002,orientation_tolerance=.025,
                velocity_scale=node.physical_test_lift_velocity_scale,
                acceleration_scale=node.physical_test_lift_velocity_scale)
    if state is not None:kwargs.update(start_state=state,open_gripper=open_gripper)
    result=node._plan_candidate(dict(position=target,orientation=q),**kwargs)
    if state is not None:
        seed=dict(zip(state.joint_state.name,state.joint_state.position))
        start=[seed[f'joint{i}'] for i in range(1,7)]+[node.gripper_open]
    else:
        with node._lock:
            start=[node._joint_positions[f'joint{i}'] for i in range(1,7)]+[node._joint_positions['jointGripper']]
    node.geometry.check_path(trajectory_points(result.planned_trajectory.joint_trajectory,start))
    return result.planned_trajectory


def preplan_lift(node,reference,state,object_position,attempt_id):
    scene=snapshot(node);created=time.monotonic()
    trajectory=plan(node,reference['position'],reference['orientation'],object_position,state,True)
    return dict(trajectory=trajectory,position=tuple(reference['position']),orientation=tuple(reference['orientation']),
                object_position=tuple(object_position),scene=scene,created=created,attempt_id=attempt_id,
                owner=getattr(node,'_grasp_candidate',None))


def execute_lift(node,attempt_id,envelope=None):
    node._raise_if_stop_requested()
    cache=getattr(node,'cached_lift',None);node.cached_lift=None
    p,q=node._current_ee_pose();origin=node._current_object_position('peach')
    reusable=False
    if cache and cache['attempt_id']==attempt_id and cache['owner'] is getattr(node,'_grasp_candidate',None):
        path=cache['trajectory'].joint_trajectory
        with node._lock:joints=dict(node._joint_positions)
        valid_joints=(bool(path.points) and len(path.points[0].positions)==len(path.joint_names)
                      and all(name in joints and abs(joints[name]-value)<=.03 for name,value in zip(path.joint_names,path.points[0].positions)))
        reusable=(0<=time.monotonic()-cache['created']<=30. and math.dist(p,cache['position'])<=.005
                  and math.dist(origin,cache['object_position'])<=.003 and angle(q,cache['orientation'])<=.035
                  and valid_joints and unchanged(node,cache['scene']))
    if reusable:
        trajectory=cache['trajectory'];node.get_logger().info('桃子纠偏：复用准备阶段预检查的试抬路径；实际起点与场景复核通过')
    else:
        trajectory=plan(node,p,q,origin)
        node.get_logger().info('桃子纠偏：试抬预检查失效，按实际夹持状态重新规划')
    if node._send_arm_trajectory(trajectory):raise RuntimeError('桃子纠偏试抬停止')
    if node._current_object_position('peach')[2] < .05:
        # Preserve the early grasp guard before the parent could attempt more lift.
        node._require_loaded_physical_grasp('peach',envelope or {},'桃子纠偏离桌后',use_verified_lift_reference=False)
    # The enclosing original _guided_grasp now performs its normal authoritative
    # consecutive probe, lift-height check and evaluator acknowledgement.
    # Do not repeat the same loaded-grasp probe inside this helper.
