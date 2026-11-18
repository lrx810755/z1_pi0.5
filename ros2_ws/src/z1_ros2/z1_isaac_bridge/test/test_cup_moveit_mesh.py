"""MoveIt cup geometry must agree with the physical cup, including root pose."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import math
import pytest
from z1_isaac_bridge.object_collision_sync import ObjectCollisionSync


def physical_geometry():
    path = Path(__file__).resolve().parents[2] / 'isaac_assets/z1/scripts/cup_collision.py'
    spec = importlib.util.spec_from_file_location('physical_cup', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cup_frustum_mesh()


def collision(name='cup', position=(.3, .1, .02), orientation=(0., 0., 0., 1.)):
    node = SimpleNamespace(world_frame='world', _object_id=lambda n: 'smolvla_object_' + n,
                           _primitive=ObjectCollisionSync._primitive)
    node._primitive = lambda n: ObjectCollisionSync._primitive(node, n)
    return ObjectCollisionSync._world_collision(node, name, position, orientation)


def test_cup_uses_cylinder_identical_to_physics_dimensions():
    from z1_isaac_bridge.cup_geometry import CUP_COLLISION
    obj = collision()
    assert not obj.meshes
    assert len(obj.primitives) == 1
    assert CUP_COLLISION['shape'] == 'cylinder'
    assert obj.primitives[0].type == obj.primitives[0].CYLINDER
    assert list(obj.primitives[0].dimensions) == pytest.approx([CUP_COLLISION['height_m'], CUP_COLLISION['radius_m']])
    assert list(obj.primitives[0].dimensions) == pytest.approx([.061,.0252])


def test_cylinder_centre_offset_rotates_with_the_cup_root():
    from z1_isaac_bridge.grasp_evaluator import rotate_vector
    q = (math.sin(.2), 0., 0., math.cos(.2))
    obj = collision(orientation=q)
    assert not obj.mesh_poses
    pose = obj.primitive_poses[0]
    offset = rotate_vector(q, (0.,0.,.061/2))
    assert (pose.position.x,pose.position.y,pose.position.z) == pytest.approx(tuple(a+b for a,b in zip((.3,.1,.02),offset)))
    assert (pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w) == q


def test_other_cylinder_is_unchanged():
    obj = collision('beverage_bottle')
    assert not obj.meshes
    assert len(obj.primitives) == 1
    assert obj.primitives[0].dimensions == pytest.approx([.157,.0242])
    assert obj.primitive_poses[0].position.z == pytest.approx(.02 + .157/2)
