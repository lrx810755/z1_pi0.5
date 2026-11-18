import sys
import json
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "isaac_assets" / "z1" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from spatial_grid import grid_cell_bounds  # noqa: E402
from layout_sweep import layout_sweep_blockers  # noqa: E402


def test_grid_cell_bounds_partition_workspace_without_gaps():
    x_bounds, y_bounds = grid_cell_bounds(
        (0.36, 0.69), (-0.27, 0.16), (5, 5), (2, 3)
    )

    assert x_bounds == pytest.approx((0.492, 0.558))
    assert y_bounds == pytest.approx((-0.012, 0.074))


@pytest.mark.parametrize("cell", [(-1, 0), (5, 0), (0, -1), (0, 5)])
def test_grid_cell_bounds_rejects_out_of_range_cell(cell):
    with pytest.raises(ValueError, match="outside grid"):
        grid_cell_bounds((0.36, 0.69), (-0.27, 0.16), (5, 5), cell)


def test_cylinder_layout_can_use_high_world_azimuth_around_box():
    config = json.loads((SCRIPTS.parents[2] / "z1_isaac_bridge" / "config" /
                         "grasp_physics.json").read_text())["scene_randomization"]
    target = (0.54, 0.12, 0.0)
    assert layout_sweep_blockers(
        "cup", target, (1.0, 0.0, 0.0, 0.0), {"cup": target}, config
    ) == {}
    blocked = (0.42, 0.12, 0.0)
    assert "smolvla_box" in layout_sweep_blockers(
        "cup", blocked, (1.0, 0.0, 0.0, 0.0), {"cup": blocked}, config
    )
