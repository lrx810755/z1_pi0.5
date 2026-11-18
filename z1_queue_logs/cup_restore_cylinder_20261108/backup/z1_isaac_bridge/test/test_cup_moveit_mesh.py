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


def test_cup_uses_identical_vertices_and_closed_triangulated_surface():
    obj = collision()
    assert not obj.primitives
    assert len(obj.meshes) == 1
    points, counts, indices = physical_geometry()
    mesh = obj.meshes[0]
    assert [(p.x, p.y, p.z) for p in mesh.vertices] == points
    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset:offset + count]
        offset += count
        triangles.extend((face[0], face[i], face[i+1]) for i in range(1, count-1))
    assert [tuple(t.vertex_indices) for t in mesh.triangles] == triangles
    edges = {}
    for triangle in mesh.triangles:
        a,b,c = triangle.vertex_indices
        for edge in ((a,b),(b,c),(c,a)):
            edges[edge] = edges.get(edge,0)+1
    assert all(n == 1 and edges.get((b,a)) == 1 for (a,b),n in edges.items())


def test_mesh_pose_is_cup_base_without_cylinder_half_height_offset():
    q = (math.sin(.2), 0., 0., math.cos(.2))
    obj = collision(orientation=q)
    assert len(obj.mesh_poses) == 1
    pose = obj.mesh_poses[0]
    assert (pose.position.x,pose.position.y,pose.position.z) == (.3,.1,.02)
    assert (pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w) == q
    assert not obj.primitive_poses


def test_other_cylinder_is_unchanged():
    obj = collision('beverage_bottle')
    assert not obj.meshes
    assert len(obj.primitives) == 1
    assert obj.primitives[0].dimensions == pytest.approx([.157,.0242])
    assert obj.primitive_poses[0].position.z == pytest.approx(.02 + .157/2)
