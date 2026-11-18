import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.rotations import rot_matrix_to_quat

from .config import RED_BALL_ENABLED, RED_BALL_PATH


def find_articulation_root(stage):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    return None


def find_prim_path_by_name(stage, prim_name):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name:
            return prim.GetPath().pathString
    return None


def prim_has_rigid_body_api(prim):
    try:
        return prim.HasAPI(UsdPhysics.RigidBodyAPI)
    except Exception:
        return False


def find_rigid_body_path(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    while prim.IsValid():
        if prim_has_rigid_body_api(prim):
            return prim.GetPath().pathString
        prim = prim.GetParent()
    return None



def look_at_quat(eye, target):
    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    forward /= forward_norm

    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up = up_hint - np.dot(up_hint, forward) * forward
    if np.linalg.norm(up) < 1e-6:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = up_hint - np.dot(up_hint, forward) * forward
    up /= np.linalg.norm(up)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    world_from_camera = np.column_stack((forward, right, up))
    return rot_matrix_to_quat(world_from_camera)


def quat_inverse(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm_sq = float(np.dot(quat, quat))
    if norm_sq == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]]) / norm_sq


def quat_multiply(lhs, rhs):
    w1, x1, y1, z1 = np.asarray(lhs, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(rhs, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_rot_matrix(quat):
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )



def force_reload_physics_from_usd():
    try:
        SimulationManager._physx_interface.force_load_physics_from_usd()
        SimulationManager._physx_interface.update_simulation(SimulationManager.get_physics_dt(), 0.0)
        SimulationManager._physx_sim_interface.fetch_results()
    except Exception as exc:
        print(f"Warning: failed to force reload PhysX USD state: {exc}")



def is_red_color(value):
    if value is None:
        return False
    if isinstance(value, (Gf.Vec3f, Gf.Vec3d)):
        colors = [value]
    else:
        try:
            colors = list(value)
        except TypeError:
            colors = [value]
    for color in colors:
        try:
            r, g, b = float(color[0]), float(color[1]), float(color[2])
        except (TypeError, IndexError, ValueError):
            continue
        if r >= 0.65 and g <= 0.35 and b <= 0.35:
            return True
    return False


def prim_has_red_material(prim):
    try:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    except Exception:
        return False
    if not material:
        return False
    surface = material.ComputeSurfaceSource()[0]
    if not surface:
        return False
    shader_prim = surface.GetPrim()
    for name in ("inputs:diffuse_color", "inputs:diffuseColor", "inputs:color"):
        attr = shader_prim.GetAttribute(name)
        if attr and is_red_color(attr.Get()):
            return True
    return False


def prim_is_red_visual(prim):
    color_attr = prim.GetAttribute("primvars:displayColor")
    return (color_attr and is_red_color(color_attr.Get())) or prim_has_red_material(prim)


def disable_prim_collision(prim):
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    except Exception:
        pass
    for child in list(prim.GetChildren()):
        disable_prim_collision(child)



def cleanup_disabled_red_targets(stage):
    if RED_BALL_ENABLED:
        return []
    hidden = []
    removed = []
    for prim in list(stage.Traverse()):
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        if path == RED_BALL_PATH or path.startswith(f"{RED_BALL_PATH}/"):
            stage.RemovePrim(prim.GetPath())
            removed.append(path)
            continue
        if not prim_is_red_visual(prim):
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        disable_prim_collision(prim)
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        hidden.append(path)

    for path in removed:
        print(f"[INFO] Removed disabled red target prim: {path}")
    for path in hidden:
        print(f"[INFO] Hid red visual prim and disabled collision: {path}")
    return removed + hidden
