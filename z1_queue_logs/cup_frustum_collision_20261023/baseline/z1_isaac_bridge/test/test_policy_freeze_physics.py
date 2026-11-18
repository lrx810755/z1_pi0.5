"""Execute the Isaac freeze method without starting the external simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('keep_collisions', [True, False])
def test_freeze_preserves_selected_collision_mode_and_fixed_pose(keep_collisions):
    source = Path(__file__).resolve().parents[2] / 'isaac_assets/z1/scripts/run_z1_ros2_scene.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SceneResetInterface')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_freeze_object')
    namespace = {'np': np, 'OBJECT_NAME_TO_PRIM': {'beverage_bottle': '/bottle'}}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    position = np.array([.5, -.1, 0.])
    orientation = np.array([1., 0., 0., 0.])
    events = []
    node = SimpleNamespace(
        _frozen_path=None, _kinematic_path=None,
        _objects={'/bottle': SimpleNamespace(get_world_pose=lambda: (position.copy(), orientation.copy()))},
        _clear_contact_state=lambda: None,
        _gripper_pose=lambda: (np.array([.3, -.1, .1]), orientation),
        _set_object_collisions=lambda path, enabled: events.append(('collision', enabled)),
        _set_object_kinematic=lambda path, enabled: events.append(('kinematic', enabled)),
        _publish_grasp_status=lambda *args, **kwargs: None,
    )
    namespace['_freeze_object'](node, 'beverage_bottle', keep_collisions=keep_collisions)
    assert events == [('collision', keep_collisions), ('kinematic', True)]
    assert node._frozen_path == '/bottle'
    assert np.array_equal(node._frozen_position, [.5, -.1, 0.])

    # A second freeze must honor a changed collision mode without moving it.
    node._frozen_collisions_enabled = keep_collisions
    namespace['_freeze_object'](node, 'beverage_bottle', keep_collisions=not keep_collisions)
    assert events[-1] == ('collision', not keep_collisions)
    assert node._frozen_collisions_enabled is (not keep_collisions)
    assert np.array_equal(node._frozen_position, [.5, -.1, 0.])

def test_contact_status_records_only_real_target_finger_penetration():
    import json
    source=Path(__file__).resolve().parents[2]/'isaac_assets/z1/scripts/run_z1_ros2_scene.py'
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SceneResetInterface')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_publish_grasp_status')
    namespace={'json':json,'GRIPPER_STATOR':'/left','GRIPPER_MOVER':'/right'}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    attributes={n.attr:None for n in ast.walk(method) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self'}
    sent=[]
    attributes.update(_target_object='beverage_bottle',_active_contact_pairs={
        'real':{'endpoint0':('/left','/left/collisions/tooth'),'endpoint1':('/bottle','/bottle/collisions'),'minimum_separation_m':-.004},
        'helper':{'endpoint0':('/left','/left/debug'),'endpoint1':('/bottle','/bottle/collisions'),'minimum_separation_m':-.05}},
        _contact_object_for_endpoint=lambda endpoint:'beverage_bottle' if endpoint[0]=='/bottle' else None,
        _endpoint_is_actual_finger_collision=lambda endpoint,root:endpoint[1].startswith(root+'/collisions/'),
        _String=lambda:SimpleNamespace(data=''),_grasp_status_pub=SimpleNamespace(publish=lambda msg:sent.append(json.loads(msg.data))))
    namespace['_publish_grasp_status'](SimpleNamespace(**attributes),'CONTACT_STATE','test')
    assert sent[0]['target_finger_min_separation_m']==-.004
    assert sent[0]['target_finger_max_penetration_m']==.004
    assert sent[0]['target_finger_contact_pair_count']==1
