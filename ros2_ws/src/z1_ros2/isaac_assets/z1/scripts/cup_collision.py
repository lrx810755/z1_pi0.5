"""Cup cylinder collider; the cup root remains at its base (Z=0).

PhysX and MoveIt take dimensions from the same shared definition.
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
    """Replace a stale mesh with the original cylinder without moving its root."""
    from pxr import UsdGeom, UsdPhysics, UsdShade

    existing = stage.GetPrimAtPath(path)
    collision_attributes = []
    if existing:
        if physics_material is None:
            physics_material = UsdShade.MaterialBindingAPI(existing).ComputeBoundMaterial(materialPurpose="physics")[0]
        collision_attributes = [(a.GetName(), a.GetTypeName(), a.Get()) for a in existing.GetAttributes()
                                if a.GetName().startswith("physxCollision:") and a.Get() is not None]
        stage.RemovePrim(path)
    radius, height = CUP_COLLISION["radius_m"], CUP_COLLISION["height_m"]
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr("Z")
    cylinder.CreateExtentAttr([(-radius, -radius, -height/2), (radius, radius, height/2)])
    cylinder.AddTranslateOp().Set((0.0, 0.0, height/2))
    prim = cylinder.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    for name, type_name, value in collision_attributes:
        prim.CreateAttribute(name, type_name).Set(value)
    if physics_material:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            physics_material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return prim


def configure_cup_collision(stage):
    """Normalize loaded assets as well as newly built scenes before physics starts."""
    root = "/World/Task/Objects/Cup"
    if stage.GetPrimAtPath(root):
        return define_cup_collider(stage, root + "/Collider")
    return None
