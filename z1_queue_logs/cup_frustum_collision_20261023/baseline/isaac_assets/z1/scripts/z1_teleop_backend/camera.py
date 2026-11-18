import struct
import threading
import time

import numpy as np
from pxr import Gf, UsdGeom

import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.rotations import rot_matrix_to_quat
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.sensors.camera import Camera

from .config import (
    CAMERA_RED_VISUAL_CLEANUP_ENABLED,
    CAMERA_VISUAL_ENABLED,
    DISPLAY_CAMERA_RESOLUTION,
    PREFERRED_WRIST_CAMERA_MOUNT,
    RED_BALL_PATH,
    RED_BALL_RADIUS_M,
    RUNTIME_WRIST_CAMERA_ENABLED,
    RUNTIME_WRIST_CAMERA_FORWARD_OFFSET_M,
    WRIST_CAMERA_FORWARD_AXIS,
    WRIST_CAMERA_FORWARD_VECTOR,
    WRIST_CAMERA_LOCAL_ROTATE_XYZ_DEG,
    WRIST_CAMERA_LOCAL_TRANSLATION,
    WRIST_CAMERA_NAME,
    WRIST_CAMERA_RESOLUTION,
)
from .safety import clamp
from .usd_utils import (
    disable_prim_collision,
    find_prim_path_by_name,
    prim_is_red_visual,
    quat_to_rot_matrix,
)


def parse_yaml_vector(text):
    return [float(item.strip()) for item in text.strip().strip("[]").split(",") if item.strip()]


def load_hand_eye_calibration(path):
    if not path.exists():
        raise FileNotFoundError(f"手眼标定文件不存在: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    calibration = {}
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        index += 1
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if key in ("T_mount_camera", "T_camera_mount", "camera_matrix_K"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("- ["):
                row_text = lines[index].strip()[2:].strip()
                rows.append(parse_yaml_vector(row_text))
                index += 1
            calibration[key] = np.asarray(rows, dtype=np.float64)
        elif value.startswith("["):
            calibration[key] = np.asarray(parse_yaml_vector(value), dtype=np.float64)
        else:
            try:
                calibration[key] = float(value)
            except ValueError:
                calibration[key] = value

    for key in ("T_mount_camera", "T_camera_mount", "camera_matrix_K"):
        if key not in calibration:
            raise RuntimeError(f"手眼标定文件缺少字段: {key}")
    return calibration



def get_camera_depth_image(camera):
    frame = camera.get_current_frame()
    for key in ("distance_to_image_plane", "distance_to_camera", "depth"):
        depth_image = frame.get(key) if isinstance(frame, dict) else None
        if depth_image is None:
            continue
        depth_image = np.asarray(depth_image, dtype=np.float32)
        if depth_image.ndim != 2:
            continue
        return depth_image
    return None


def depth_from_camera(camera, detection, calibration):
    pixel = np.round(detection["pixel"]).astype(int)
    u = int(clamp(pixel[0], 0, WRIST_CAMERA_RESOLUTION[0] - 1))
    v = int(clamp(pixel[1], 0, WRIST_CAMERA_RESOLUTION[1] - 1))
    depth_image = get_camera_depth_image(camera)
    if depth_image is not None:
        y0 = max(v - 2, 0)
        y1 = min(v + 3, depth_image.shape[0])
        x0 = max(u - 2, 0)
        x1 = min(u + 3, depth_image.shape[1])
        patch = depth_image[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 3.0)]
        if valid.size:
            return float(np.median(valid)), "distance_to_image_plane"

    fx = float(calibration["camera_matrix_K"][0, 0])
    radius_px = max(float(detection["radius_px"]), 1.0)
    return float(fx * RED_BALL_RADIUS_M / radius_px), "radius_estimate"


def camera_point_from_pixel_depth(pixel, depth_m, calibration):
    fx = float(calibration["camera_matrix_K"][0, 0])
    fy = float(calibration["camera_matrix_K"][1, 1])
    cx = float(calibration["camera_matrix_K"][0, 2])
    cy = float(calibration["camera_matrix_K"][1, 2])
    u, v = np.asarray(pixel, dtype=np.float64)

    x_right = (u - cx) / fx * depth_m
    y_down = (v - cy) / fy * depth_m
    return np.array([x_right, -y_down, -depth_m, 1.0], dtype=np.float64)


def transform_camera_point_to_world(camera_point, calibration, mount_path):
    mount_point = calibration["T_mount_camera"] @ camera_point
    mount_position, mount_quat = get_world_pose(mount_path)
    mount_rotation = quat_to_rot_matrix(mount_quat)
    world_point = np.asarray(mount_position, dtype=np.float64) + mount_rotation @ mount_point[:3]
    return mount_point[:3], world_point



def rgba_to_bmp(rgba):
    image = np.asarray(rgba)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("camera frame must be HxWxRGBA/RGB")
    rgb = image[:, :, :3]
    if rgb.dtype != np.uint8:
        if rgb.max(initial=0.0) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    height, width, _ = rgb.shape
    row_stride = ((width * 3 + 3) // 4) * 4
    padding = b"\x00" * (row_stride - width * 3)
    pixel_rows = []
    for row in rgb[::-1]:
        pixel_rows.append(row[:, ::-1].tobytes() + padding)
    pixel_data = b"".join(pixel_rows)
    file_size = 14 + 40 + len(pixel_data)
    header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
    dib = struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, len(pixel_data), 2835, 2835, 0, 0)
    return header + dib + pixel_data



def camera_mount_orientation(forward_axis, up_axis):
    """Return camera orientation relative to the mount frame.

    USD/Isaac Camera convention:
      camera local -Z = optical/view direction
      camera local +Y = image up
      camera local +X = image right

    This function aligns camera local -Z to a chosen mount-frame axis, so the
    camera view follows the gripper/link direction instead of looking at a
    fixed world-space target.
    """
    forward = np.asarray(forward_axis, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-9)

    up_hint = np.asarray(up_axis, dtype=np.float64)
    up_hint /= max(np.linalg.norm(up_hint), 1e-9)

    up = up_hint - np.dot(up_hint, forward) * forward
    if np.linalg.norm(up) < 1e-6:
        # Pick a stable fallback if forward and up are parallel.
        for candidate in (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        ):
            up = candidate - np.dot(candidate, forward) * forward
            if np.linalg.norm(up) >= 1e-6:
                break
    up /= max(np.linalg.norm(up), 1e-9)

    camera_z = -forward
    camera_y = up
    camera_x = np.cross(camera_y, camera_z)
    camera_x /= max(np.linalg.norm(camera_x), 1e-9)
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= max(np.linalg.norm(camera_y), 1e-9)

    mount_from_camera = np.column_stack((camera_x, camera_y, camera_z))
    return rot_matrix_to_quat(mount_from_camera)



def cleanup_camera_red_visuals(stage, camera_path, mount_path):
    if not CAMERA_RED_VISUAL_CLEANUP_ENABLED:
        return []

    camera_prefix = f"{camera_path}/"
    mount_prefix = f"{mount_path}/"
    removed = []
    hidden = []
    hints = ("camera", "wrist", "visible", "calib", "checker", "target", "red")

    for prim in list(stage.Traverse()):
        path = prim.GetPath().pathString
        if path == RED_BALL_PATH or path.startswith(f"{RED_BALL_PATH}/"):
            continue
        if not path.startswith((camera_prefix, mount_prefix)):
            continue
        path_lower = path.lower()
        if not prim_is_red_visual(prim) and not any(hint in path_lower for hint in hints):
            continue
        if prim.IsA(UsdGeom.Camera):
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        disable_prim_collision(prim)
        if path.startswith(camera_prefix):
            stage.RemovePrim(prim.GetPath())
            removed.append(path)
        else:
            imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            hidden.append(path)

    for path in removed:
        print(f"[INFO] Removed camera obstruction prim: {path}")
    for path in hidden:
        print(f"[INFO] Hid camera obstruction prim and disabled collision: {path}")
    return removed + hidden



def create_camera_visual(stage, camera_path):
    """Create a tiny non-collision white camera body under the Camera prim."""
    body_path = f"{camera_path}/VisibleCameraBody"
    body_prim = stage.GetPrimAtPath(body_path)
    if body_prim.IsValid():
        body = UsdGeom.Cube(body_prim)
    else:
        body = UsdGeom.Cube.Define(stage, body_path)
    body.GetDisplayColorAttr().Set([Gf.Vec3f(0.92, 0.92, 0.88)])
    xformable = UsdGeom.Xformable(body.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xformable.AddScaleOp().Set(Gf.Vec3f(0.012, 0.010, 0.008))


def configure_camera_prim_xform(camera_prim):
    xformable = UsdGeom.Xformable(camera_prim)
    xformable.ClearXformOpOrder()
    translation = np.asarray(WRIST_CAMERA_LOCAL_TRANSLATION, dtype=np.float64).copy()
    if RUNTIME_WRIST_CAMERA_FORWARD_OFFSET_M:
        translation += WRIST_CAMERA_FORWARD_VECTOR * float(RUNTIME_WRIST_CAMERA_FORWARD_OFFSET_M)
    xformable.AddTranslateOp().Set(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
    xformable.AddRotateXYZOp().Set(
        Gf.Vec3f(
            float(WRIST_CAMERA_LOCAL_ROTATE_XYZ_DEG[0]),
            float(WRIST_CAMERA_LOCAL_ROTATE_XYZ_DEG[1]),
            float(WRIST_CAMERA_LOCAL_ROTATE_XYZ_DEG[2]),
        )
    )


def copy_camera_prim_settings(source_prim, target_camera):
    target_prim = target_camera.GetPrim()
    configure_camera_prim_xform(target_prim)
    for attr_name in (
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "clippingRange",
        "focusDistance",
        "fStop",
    ):
        value = source_prim.GetAttribute(attr_name).Get()
        if value is not None:
            target_prim.GetAttribute(attr_name).Set(value)


def hide_camera_visual_children(stage, camera_path):
    camera_prim = stage.GetPrimAtPath(camera_path)
    hidden = []
    for child in list(camera_prim.GetChildren()):
        if child.IsA(UsdGeom.Camera):
            continue
        disable_prim_collision(child)
        imageable = UsdGeom.Imageable(child)
        if imageable:
            imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        try:
            stage.RemovePrim(child.GetPath())
        except Exception:
            pass
        hidden.append(child.GetPath().pathString)
    for path in hidden:
        print(f"[INFO] Disabled original wrist camera visual child: {path}")
    return hidden


def create_wrist_camera(stage):
    mount_candidates = []
    if PREFERRED_WRIST_CAMERA_MOUNT:
        mount_candidates.append(PREFERRED_WRIST_CAMERA_MOUNT)
    mount_candidates.extend(["gripperStator", "link06", "gripperMover"])

    mount_path = None
    mount_name = None
    for candidate in mount_candidates:
        mount_path = find_prim_path_by_name(stage, candidate)
        if mount_path is not None:
            mount_name = candidate
            break
    if mount_path is None:
        raise RuntimeError("未找到 gripperStator、link06 或 gripperMover，无法安装腕部摄像头。")

    source_camera_path = f"{mount_path}/{WRIST_CAMERA_NAME}"
    camera_prim = stage.GetPrimAtPath(source_camera_path)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(
            "USD 中未找到腕部 Camera prim。请先运行: "
            "$ISAAC_SIM_ROOT/python.sh "
            "$Z1_ASSET_ROOT/scripts/insert_wrist_camera_usd.py"
        )
    cleanup_camera_red_visuals(stage, source_camera_path, mount_path)
    hide_camera_visual_children(stage, source_camera_path)
    configure_camera_prim_xform(camera_prim)

    camera_path = source_camera_path
    if RUNTIME_WRIST_CAMERA_ENABLED:
        camera_path = f"{mount_path}/{WRIST_CAMERA_NAME}Runtime"
        runtime_camera = UsdGeom.Camera.Define(stage, camera_path)
        copy_camera_prim_settings(camera_prim, runtime_camera)
        hide_camera_visual_children(stage, camera_path)
        print(f"[INFO] Runtime wrist camera created: {camera_path}")

    if CAMERA_VISUAL_ENABLED:
        create_camera_visual(stage, camera_path)

    camera = Camera(
        prim_path=camera_path,
        name="wrist_camera",
        frequency=20,
        resolution=WRIST_CAMERA_RESOLUTION,
    )
    camera.initialize()
    try:
        camera.add_distance_to_image_plane_to_frame()
    except Exception as exc:
        print(f"[WARN] 腕部相机深度帧启用失败，将使用红球半径估算距离: {exc}")
    print(f"[INFO] USD wrist camera mount selected by name: {mount_name}")
    print(
        "[INFO] Wrist camera local pose: "
        f"translation={np.round(WRIST_CAMERA_LOCAL_TRANSLATION, 4).tolist()} "
        f"rotateXYZ_deg={np.round(WRIST_CAMERA_LOCAL_ROTATE_XYZ_DEG, 2).tolist()} "
        f"optical_axis_mount={WRIST_CAMERA_FORWARD_AXIS}"
    )
    return camera, camera_path, mount_path


def create_display_camera():
    camera = Camera(
        prim_path="/World/TeleopDisplayCamera",
        name="teleop_display_camera",
        position=np.array([0.3, 0.0, 1.45]),
        orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True),
        frequency=20,
        resolution=DISPLAY_CAMERA_RESOLUTION,
    )
    camera.initialize()
    return camera


class CameraFrameStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_time = 0.0
        self.error = "摄像头画面尚未准备好"

    def update(self, camera):
        try:
            rgba = camera.get_rgba()
            if rgba is None:
                self.set_error("摄像头画面尚未准备好")
                return
            frame = rgba_to_bmp(rgba)
        except Exception as exc:
            self.set_error(f"摄像头画面更新失败: {exc}")
            return
        with self.lock:
            self.frame = frame
            self.frame_time = time.time()
            self.error = ""

    def set_error(self, error):
        with self.lock:
            self.error = error

    def snapshot(self):
        with self.lock:
            return self.frame, self.frame_time, self.error
