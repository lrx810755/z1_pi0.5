import shutil
from pathlib import Path

from isaacsim import SimulationApp

from project_paths import ASSET_ROOT

print("Starting USD wrist camera insertion.", flush=True)
simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom


USD_PATH = ASSET_ROOT / "z1.usd"
BACKUP_PATH = ASSET_ROOT / "z1.usd.before_wrist_camera"
LOG_PATH = Path("/tmp/z1_insert_wrist_camera_usd.log")
MOUNT_NAME = "gripperStator"
CAMERA_NAME = "WristCamera"
# gripperStator/gripperMover geometry extends mostly along local +X.
# Mount the eye-in-hand camera on the fixed gripper base, outside the finger
# volume, with optical axis along local +X.
CAMERA_LOCAL_TRANSLATION = (0.055, -0.058, 0.035)
CAMERA_LOCAL_ROTATE_XYZ_DEG = (90.0, 0.0, -90.0)
CAMERA_RESOLUTION = (640, 360)


def log(message):
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")
    print(message, flush=True)


def find_prim_path_by_name(stage, prim_name):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name:
            return prim.GetPath().pathString
    return None


def set_xform(prim, translation, scale=None):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def set_display_color(prim, color):
    imageable = UsdGeom.Gprim(prim)
    imageable.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def define_visible_cube(stage, path, translation, scale, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_xform(cube.GetPrim(), translation, scale)
    set_display_color(cube.GetPrim(), color)
    return cube


try:
    LOG_PATH.write_text("", encoding="utf-8")
    log("enter try")
    if not USD_PATH.exists():
        raise FileNotFoundError(f"USD file does not exist: {USD_PATH}")

    if not BACKUP_PATH.exists():
        shutil.copy2(USD_PATH, BACKUP_PATH)

    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {USD_PATH}")
    root_layer = stage.GetRootLayer()
    stage.SetEditTarget(root_layer)
    log(f"edit target: {stage.GetEditTarget().GetLayer().identifier}")

    mount_path = find_prim_path_by_name(stage, MOUNT_NAME)
    if mount_path is None:
        raise RuntimeError(f"Could not find mount prim named {MOUNT_NAME}")
    log(f"mount path: {mount_path}")

    camera_path = f"{mount_path}/{CAMERA_NAME}"
    if stage.GetPrimAtPath(camera_path).IsValid():
        if not stage.RemovePrim(camera_path):
            raise RuntimeError(f"Failed to remove existing camera prim: {camera_path}")
        log(f"removed existing camera: {camera_path}")

    camera = UsdGeom.Camera.Define(stage, camera_path)
    log(f"defined camera: {camera_path}")
    camera_prim = camera.GetPrim()

    xform = UsdGeom.Xformable(camera_prim)
    log("created xformable")
    xform.ClearXformOpOrder()
    log("cleared xform order")
    xform.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_LOCAL_TRANSLATION))
    log("set translate")
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*CAMERA_LOCAL_ROTATE_XYZ_DEG))
    log("set rotateXYZ")

    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
    log("set clipping")
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.71625)
    camera.CreateFocalLengthAttr(24.0)
    define_visible_cube(
        stage,
        f"{camera_path}/VisibleCameraBody",
        (0.0, 0.0, 0.0),
        (0.018, 0.014, 0.012),
        (0.02, 0.02, 0.02),
    )
    define_visible_cube(
        stage,
        f"{camera_path}/VisibleCameraLens",
        (0.0, 0.0, -0.020),
        (0.010, 0.008, 0.006),
        (1.0, 0.0, 0.0),
    )
    define_visible_cube(
        stage,
        f"{camera_path}/VisibleCameraBack",
        (0.0, 0.0, 0.018),
        (0.008, 0.008, 0.004),
        (1.0, 0.85, 0.0),
    )

    print(f"Root layer dirty before save: {root_layer.dirty}", flush=True)
    log(f"dirty before save: {root_layer.dirty}")
    if not root_layer.Save():
        raise RuntimeError(f"Failed to save root layer: {root_layer.identifier}")
    log(f"root layer saved: {root_layer.identifier}")

    verify_stage = Usd.Stage.Open(str(USD_PATH))
    verify_prim = verify_stage.GetPrimAtPath(camera_path) if verify_stage else None
    if not verify_prim or not verify_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Saved USD does not contain Camera prim: {camera_path}")
    verify_matrix = UsdGeom.Xformable(verify_prim).GetLocalTransformation()
    verify_translation = verify_matrix.ExtractTranslation()
    if abs(verify_translation[0] - CAMERA_LOCAL_TRANSLATION[0]) > 1e-6:
        raise RuntimeError(f"Camera translation was not saved: {verify_translation}")
    if not verify_stage.GetPrimAtPath(f"{camera_path}/VisibleCameraBody").IsValid():
        raise RuntimeError("Saved USD does not contain VisibleCameraBody")

    log(f"Inserted USD wrist camera: {camera_path}")
    log(f"Backup: {BACKUP_PATH}")
finally:
    simulation_app.close()
