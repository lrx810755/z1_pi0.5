from types import SimpleNamespace
import pytest
from z1_isaac_bridge import object_collision_sync as sync

def test_peach_proxy_reads_configured_radius_and_base_relative_center(monkeypatch):
    proxy=dict(sync.GRASP_CONFIG['collision_proxy']['peach'],planning_radius_m=.0312,planning_center_z_m=.0174)
    monkeypatch.setitem(sync.GRASP_CONFIG['collision_proxy'],'peach',proxy)
    node=SimpleNamespace()
    primitive,offset=sync.ObjectCollisionSync._primitive(node,'peach')
    assert primitive.dimensions == pytest.approx([.0312])
    assert offset == pytest.approx(.0174)
