"""Cup corrections must match the base grid and command-aligned physical labels."""
import json
import importlib.util
from pathlib import Path
from collections import Counter
import pytest

HERE=Path(__file__).resolve().parent

def local(name):
    p=HERE/(name+'.py')
    assert p.exists(), f'missing cup {name}'
    spec=importlib.util.spec_from_file_location('cup_'+name,p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_sixty_tasks_cover_all_twenty_base_cells():
    c=local('core');tasks=c.make_tasks()
    assert len(tasks)==60 and len({t['id'] for t in tasks})==60
    counts=Counter(tuple(t['grid']) for t in tasks)
    assert set(counts)=={(col,row) for row in range(4) for col in range(5)}
    assert set(counts.values())=={3}
    for t in tasks:
        assert t['id'].startswith('cup_')
        assert (t['split']=='validation') == (tuple(t['grid']) in {(1,1),(2,3)})
        assert abs(t['offset_approach_m'][2])<=.002
        assert max(map(abs,t['rotation_tool_deg']))<=3


def test_open_command_transition_does_not_depend_on_grasping_phase():
    f=local('fragment_core')
    rows=[dict(action=[0]*6+[x],phase='READY') for x in [-.9]*3+[-.88]*17]
    labels,close,last=f.partial_labels(rows)
    assert close==3 and labels==[0]*3+[1]*17 and last==4
    with pytest.raises(ValueError):f.partial_labels(rows+[dict(action=[0]*6+[-.9])])


def test_fragment_requires_dynamic_real_cup_grasp():
    f=local('fragment_core')
    s=dict(target_object='cup',object_position_m=[.5,0,.06],physical_grasp_ok=True,lifted=True,
           bilateral_contact=True,target_collisions_enabled=True,frozen=False,object_touching_table=False,
           left_finger_touching_table=False,right_finger_touching_table=False)
    assert f.valid_hold(s)
    for key,value in [('target_object','beverage_bottle'),('bilateral_contact',False),('frozen',True),('object_position_m',[.5,0,.049])]:
        assert not f.valid_hold(dict(s,**{key:value}))


def test_bottle_or_incomplete_episode_never_counts_as_cup(tmp_path):
    c=local('core');p=tmp_path/'episode_000000';p.mkdir()
    (p/'correction.json').write_text(json.dumps(dict(accepted=True,plan_hash='x',task_id='cup_left_001')))
    (p/'frames.jsonl').write_text('{}\n')
    summary=dict(fragment_success=True,physical_grasp_ok=True,target_object='beverage_bottle')
    (p/'episode_summary.json').write_text(json.dumps(summary))
    assert c.completed_ids(tmp_path,'x')==set()
    summary['target_object']='cup';(p/'episode_summary.json').write_text(json.dumps(summary))
    assert c.completed_ids(tmp_path,'x')=={'cup_left_001'}


def test_offsets_follow_actual_horizontal_tool_approach():
    p=local('preflight')
    # Tool X points world +Y here: moving left means world -X.
    import math
    q=[0,0,math.sin(math.pi/4),math.cos(math.pi/4)]
    task=dict(kind='left',offset_approach_m=[0,.006,0],rotation_tool_deg=[0,0,0])
    r=next(p.routes([.4,0,.1],q,[.5,0,0],task))
    assert r['bad_position'][0]==pytest.approx(.394)
    assert r['bad_position'][2]==pytest.approx(.1)
    assert r['preparation'][0]['linear'] is False


def test_preparation_and_recovery_are_planned_before_recording(monkeypatch):
    import ast,copy,threading,time,sys,types,numpy as np
    from types import SimpleNamespace as NS
    from scipy.spatial.transform import Rotation
    p=local('preflight');c=local('core')
    tree=ast.parse((HERE/'nodes.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_generate_plan')
    class Base:
        def _generate_plan(self):return 'approved'
    moveit=types.ModuleType('moveit_msgs.msg')
    moveit.RobotState=lambda:NS(is_diff=True,joint_state=NS(name=[],position=[]))
    env=dict(Base=Base,copy=copy,time=time,Rotation=Rotation,trajectory_points=p.trajectory_points,
             routes=p.routes,choose_route=p.choose_route,context=lambda:dict(task=c.make_tasks()[0]))
    klass=ast.ClassDef(name='Subject',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),'nodes.py','exec'),env)
    x=env['Subject']();x._lock=threading.Lock();x._joint_positions={f'joint{i}':0. for i in range(1,7)};x.gripper_open=-.9
    approved=NS(joint_trajectory=NS(joint_names=[f'joint{i}' for i in range(1,7)],points=[NS(positions=[0,-.5,-1.,0,0,0])]))
    x._approved_trajectory=approved;x.geometry=NS(link_fk=lambda q:(np.array([.5,0,.15]),np.eye(3)),check_path=lambda pts:None)
    x._current_object_position=lambda _: [.5,0,0];x._raise_if_stop_requested=lambda:None
    starts=[]
    def plan(candidate,**kwargs):
        starts.append(list(kwargs['start_state'].joint_state.position))
        return NS(planned_trajectory=approved)
    x._plan_candidate=plan
    x._state_after_trajectory=lambda result:NS(is_diff=True,joint_state=NS(name=[f'joint{i}' for i in range(1,7)]+['jointGripper'],position=[0,-.5,-1.,0,0,0,-.9]))
    monkeypatch.setitem(sys.modules,'moveit_msgs.msg',moveit)
    x._generate_plan()
    route,planned,_=x.cached
    assert starts[0][2]==0 and len(planned)==len(route['preparation'])+len(route['recovery'])
    assert len(planned)==7 and route['preparation'][0]['linear'] is False
