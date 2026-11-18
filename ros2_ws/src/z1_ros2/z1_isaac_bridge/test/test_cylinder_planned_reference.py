"""Check the actual final-pose calculation against Isaac's center probe."""
import ast
import inspect
import math
import textwrap
from types import SimpleNamespace

import pytest
from z1_isaac_bridge import grasp_assistant as module


@pytest.mark.parametrize("target,roll", [("beverage_bottle", 90),
                                         ("beverage_bottle", -90), ("cup", 120)])
def test_exact_planned_pose_passes_center_based_acceptance(target, roll):
    # Execute the real geometry-to-final-pose block without starting MoveIt.
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.GraspAssistant._generate_plan)))
    statements = None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        start = next((i for i, item in enumerate(body)
                      if isinstance(item, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "geometry_point"
                              for t in item.targets)), None)
        if start is not None:
            end = next(i for i in range(start, len(body))
                       if isinstance(body[i], ast.Assign)
                       and any(isinstance(t, ast.Subscript)
                               and isinstance(t.slice, ast.Constant)
                               and t.slice.value == "standoff_m" for t in body[i].targets))
            statements = body[start:end + 1]
            break
    assert statements is not None
    profile = module.GRASP_PROFILES[target]
    root = (0.38, -0.19, 0.004)
    orientation = module.orientation_for_approach((1.0, 0.0, -0.0875), roll)
    rotation = SimpleNamespace(x=0.02, y=0.0, z=0.0, w=math.sqrt(1 - 0.02 ** 2))
    translation = SimpleNamespace(x=0.0, y=0.003, z=0.01)
    geometry = (root[0], root[1], root[2] + profile["height"]
                + profile.get("world_z_offset_m", 0.0))
    desired = (0.12283045, -0.02663304, 0.0187797)
    candidate = {"target_point": geometry, "orientation": orientation}
    final = dict(candidate)
    context = dict(vars(module), candidate=candidate, final_candidate=final,
                   profile=profile, object_position=root, planning_target=target,
                   rotation=rotation, translation=translation, desired_local=desired)
    exec(compile(ast.Module(body=statements, type_ignores=[]), "final_pose", "exec"), context)
    stator_q = module.quaternion_multiply(orientation,
                                         (rotation.x, rotation.y, rotation.z, rotation.w))
    offset = module.rotate_vector(orientation, (translation.x, translation.y, translation.z))
    stator_position = tuple(final["position"][i] + offset[i] for i in range(3))
    center = (root[0], root[1], root[2] + module.GRASP_CONFIG["object_center_z_offsets_m"][target])
    measured = module.rotate_vector(module.quaternion_conjugate(stator_q),
                                    tuple(center[i] - stator_position[i] for i in range(3)))
    assert final["desired_object_in_stator_m"] == pytest.approx(measured, abs=1e-10)
    # Converting acceptance must leave the planned physical grasp pose intact.
    planned_point = (root[0], root[1], root[2] + profile.get(
        "alignment_geometry_z_offset_m", geometry[2] - root[2]))
    assert final["position"] == pytest.approx(module._ee_position_for_object_in_stator(
        planned_point, orientation, (translation.x, translation.y, translation.z),
        (rotation.x, rotation.y, rotation.z, rotation.w), desired))
