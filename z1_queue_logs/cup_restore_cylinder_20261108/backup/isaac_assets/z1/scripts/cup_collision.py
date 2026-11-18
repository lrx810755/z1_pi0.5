"""Solid tapered cup collider, measured from the normalized scanned visual.

The cup root is at the base (Z=0). PhysX and MoveIt use the same
shared frustum geometry.
"""

import importlib.util
from pathlib import Path

# Load the shared pure geometry without requiring ROS packages in Isaac Python.
_geometry_path = Path(__file__).resolve().parents[3] / "z1_isaac_bridge/z1_isaac_bridge/cup_geometry.py"
_spec = importlib.util.spec_from_file_location("z1_shared_cup_geometry", _geometry_path)
_geometry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geometry)
CUP_COLLISION = _geometry.CUP_COLLISION
cup_frustum_mesh = _geometry.cup_frustum_mesh


def define_cup_collider(stage, path, physics_material=None):
    """Author the same PhysX-compatible collider for builds and live assets."""
    from pxr import UsdGeom, UsdPhysics, UsdShade

    points, counts, indices = cup_frustum_mesh()
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    radius = CUP_COLLISION["top_radius_m"]
    mesh.CreateExtentAttr([(-radius, -radius, 0.0), (radius, radius, CUP_COLLISION["height_m"])])
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    if physics_material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            physics_material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return prim
