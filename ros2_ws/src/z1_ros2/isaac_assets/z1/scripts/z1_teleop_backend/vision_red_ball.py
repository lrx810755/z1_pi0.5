import threading

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics

try:
    import cv2
except Exception:
    cv2 = None

from isaacsim.core.api.materials import OmniPBR, PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere, FixedCuboid
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.xforms import get_world_pose

from .camera import get_camera_depth_image
from .config import (
    GRASP_ASSISTED_ATTACHMENT_ENABLED,
    GRASP_ATTACH_MAX_DISTANCE_M,
    GRASP_USE_PHYSICS_FIXED_JOINT,
    RED_BALL_DEFAULT_POSITION,
    RED_BALL_DISABLE_GRAVITY,
    RED_BALL_ENABLED,
    RED_BALL_EDGE_MARGIN_PX,
    RED_BALL_MASS_KG,
    RED_BALL_MIN_AREA_PX,
    RED_BALL_MIN_CIRCULARITY,
    RED_BALL_MIN_RADIUS_PX,
    RED_BALL_PATH,
    RED_BALL_RADIUS_M,
    FIXED_DEMO_BALL_POSITION,
    FIXED_DEMO_SUPPORT_ENABLED,
    FIXED_DEMO_SUPPORT_PATH,
    FIXED_DEMO_SUPPORT_SIZE_M,
    RED_OBJECT_GEOMETRY,
    RED_OBJECT_SIZE_M,
)
from .safety import clamp
from .usd_utils import (
    cleanup_disabled_red_targets,
    find_rigid_body_path,
    force_reload_physics_from_usd,
    quat_inverse,
    quat_multiply,
    quat_to_rot_matrix,
)


def detect_red_ball(camera):
    rgba = camera.get_rgba()
    if rgba is None:
        return {"found": False, "red_pixels": 0, "reason": "camera_frame_missing"}
    rgb = np.asarray(rgba[:, :, :3])
    if rgb.dtype != np.uint8:
        rgb_float = rgb.astype(np.float32)
        if rgb_float.max(initial=0.0) <= 1.0:
            rgb_float *= 255.0
        rgb = np.clip(rgb_float, 0, 255).astype(np.uint8)

    if cv2 is None:
        red = rgb[:, :, 0].astype(np.float32)
        green = rgb[:, :, 1].astype(np.float32)
        blue = rgb[:, :, 2].astype(np.float32)
        dominance = red - np.maximum(green, blue)
        mask = ((red > 80.0) & (dominance > 25.0)).astype(np.uint8) * 255
    else:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask_low = cv2.inRange(hsv, np.array([0, 70, 55]), np.array([12, 255, 255]))
        mask_high = cv2.inRange(hsv, np.array([168, 70, 55]), np.array([179, 255, 255]))
        mask = cv2.bitwise_or(mask_low, mask_high)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    depth_image = get_camera_depth_image(camera)
    if depth_image is not None:
        valid_depth = np.isfinite(depth_image) & (depth_image > 0.05) & (depth_image < 3.0)
        mask = np.where(valid_depth, mask, 0).astype(np.uint8)
    red_pixels = int(np.count_nonzero(mask))

    if cv2 is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) >= float(RED_BALL_MIN_AREA_PX)]
        if not contours:
            return {"found": False, "red_pixels": red_pixels, "reason": "no_red_contour"}
        contour = max(contours, key=cv2.contourArea)
        area_px = float(cv2.contourArea(contour))
        perimeter_px = float(cv2.arcLength(contour, True))
        circularity = 4.0 * np.pi * area_px / max(perimeter_px * perimeter_px, 1e-9)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            return {"found": False, "red_pixels": red_pixels, "reason": "invalid_contour_moments"}
        u = float(moments["m10"] / moments["m00"])
        v = float(moments["m01"] / moments["m00"])
        x, y, width, height = cv2.boundingRect(contour)
        (_, _), radius_px = cv2.minEnclosingCircle(contour)
        image_height, image_width = mask.shape[:2]
        complete = (
            radius_px >= RED_BALL_MIN_RADIUS_PX
            and circularity >= RED_BALL_MIN_CIRCULARITY
            and x > RED_BALL_EDGE_MARGIN_PX
            and y > RED_BALL_EDGE_MARGIN_PX
            and x + width < image_width - RED_BALL_EDGE_MARGIN_PX
            and y + height < image_height - RED_BALL_EDGE_MARGIN_PX
        )
        return {
            "found": True,
            "pixel": np.array([u, v], dtype=np.float64),
            "red_pixels": red_pixels,
            "area_px": int(area_px),
            "bbox": [int(x), int(y), int(x + width - 1), int(y + height - 1)],
            "radius_px": float(radius_px),
            "circularity": float(circularity),
            "complete": bool(complete),
        }

    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return {"found": False, "red_pixels": red_pixels, "reason": "too_few_red_pixels"}

    u = float(xs.mean())
    v = float(ys.mean())
    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    radius_px = 0.25 * ((max_x - min_x + 1) + (max_y - min_y + 1))
    image_height, image_width = mask.shape[:2]
    complete = (
        radius_px >= RED_BALL_MIN_RADIUS_PX
        and min_x > RED_BALL_EDGE_MARGIN_PX
        and min_y > RED_BALL_EDGE_MARGIN_PX
        and max_x < image_width - RED_BALL_EDGE_MARGIN_PX
        and max_y < image_height - RED_BALL_EDGE_MARGIN_PX
    )
    return {
        "found": True,
        "pixel": np.array([u, v], dtype=np.float64),
        "red_pixels": red_pixels,
        "area_px": int(len(xs)),
        "bbox": [min_x, min_y, max_x, max_y],
        "radius_px": float(radius_px),
        "circularity": None,
        "complete": bool(complete),
    }



class RedBallManager:
    def __init__(self, world_obj, prim_path=RED_BALL_PATH):
        self.world = world_obj
        self.prim_path = prim_path
        self.grasp_joint_path = "/World/CameraTargetGraspFixedJoint"
        self.attached_body_path = None
        self.attachment_mode = "free"
        self.assisted_local_pos = None
        self.assisted_local_rot = None
        self.last_attach_distance = None
        self.last_attach_body_origin_distance = None
        self.lock = threading.Lock()
        self.default_position = RED_BALL_DEFAULT_POSITION.copy()
        self.zero_velocity = np.zeros(3, dtype=np.float64)
        self.physics_material = PhysicsMaterial(
            prim_path="/World/Materials/CameraTargetPhysicsMaterial",
            name="camera_target_physics_material",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.05,
        )
        self.geometry = RED_OBJECT_GEOMETRY
        self.object_size = np.asarray(RED_OBJECT_SIZE_M, dtype=np.float64)
        self.ball = self.world.scene.add(self._create_target_object())
        self._configure_ball_rigid_body()
        self.attached = False
        self.pending_reset = None

    def _create_target_object(self):
        material = OmniPBR(
            prim_path="/World/Materials/CameraTargetOmniPBR",
            name="camera_target_omni_pbr",
            color=np.array([1.0, 0.0, 0.0]),
        )
        if self.geometry in ("box", "cube", "cuboid"):
            return DynamicCuboid(
                prim_path=self.prim_path,
                name="camera_target",
                position=self.default_position,
                scale=self.object_size,
                mass=RED_BALL_MASS_KG,
                color=np.array([1.0, 0.0, 0.0]),
                visual_material=material,
                physics_material=self.physics_material,
            )
        if self.geometry not in ("sphere", "ball"):
            print(f"[WARN] Z1_RED_OBJECT_GEOMETRY={self.geometry!r} 无效，使用 sphere。")
            self.geometry = "sphere"
        return DynamicSphere(
            prim_path=self.prim_path,
            name="camera_target",
            position=self.default_position,
            radius=RED_BALL_RADIUS_M,
            mass=RED_BALL_MASS_KG,
            color=np.array([1.0, 0.0, 0.0]),
            visual_material=material,
            physics_material=self.physics_material,
        )

    def _configure_ball_rigid_body(self):
        prim = self.world.stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid_api.CreateDisableGravityAttr().Set(bool(RED_BALL_DISABLE_GRAVITY))
        if RED_BALL_DISABLE_GRAVITY and hasattr(self.ball, "_rigid_prim_view"):
            try:
                self.ball._rigid_prim_view.disable_gravities()
            except Exception as exc:
                print(f"Warning: failed to disable red ball gravity through runtime view: {exc}")

    def configure_runtime_physics(self):
        with self.lock:
            self._configure_ball_rigid_body()
            self._stop_ball_locked()

    def _remove_grasp_joint_locked(self):
        if self.world.stage.GetPrimAtPath(self.grasp_joint_path).IsValid():
            self.world.stage.RemovePrim(self.grasp_joint_path)

    def _stop_ball_locked(self):
        self.ball.set_linear_velocity(self.zero_velocity)
        self.ball.set_angular_velocity(self.zero_velocity)

    def request_reset(self, position=None):
        with self.lock:
            target = self.default_position if position is None else np.asarray(position, dtype=np.float64)
            if target.shape != (3,):
                raise ValueError("红球位置必须是长度为 3 的 xyz 数组")
            self.pending_reset = target.copy()
            self._remove_grasp_joint_locked()
            self.attached = False
            self.attached_body_path = None
            self.attachment_mode = "free"
            self.assisted_local_pos = None
            self.assisted_local_rot = None
            self.last_attach_distance = None
            self.last_attach_body_origin_distance = None
            snapshot = self.snapshot_locked()
            snapshot["reset_pending"] = True
            return snapshot

    def apply_pending_reset(self):
        with self.lock:
            if self.pending_reset is None:
                return None
            target = self.pending_reset.copy()
            self.pending_reset = None
            self._remove_grasp_joint_locked()
            self.ball.set_world_pose(position=target)
            self._stop_ball_locked()
            self._configure_ball_rigid_body()
            self.attached = False
            self.attached_body_path = None
            self.attachment_mode = "free"
            self.assisted_local_pos = None
            self.assisted_local_rot = None
            self.last_attach_distance = None
            self.last_attach_body_origin_distance = None
            return self.snapshot_locked()

    def hold_world_pose(self, position):
        with self.lock:
            target = np.asarray(position, dtype=np.float64)
            if target.shape != (3,):
                raise ValueError("目标物体保持位置必须是长度为 3 的 xyz 数组")
            if self.attached:
                return self.snapshot_locked()
            self.ball.set_world_pose(position=target)
            self._stop_ball_locked()
            return self.snapshot_locked()

    def attach_to_physics_joint(self, body_paths, expected_ball_position=None, force=False):
        if not GRASP_USE_PHYSICS_FIXED_JOINT and not force:
            return self.snapshot()
        with self.lock:
            ball_prim = self.world.stage.GetPrimAtPath(self.prim_path)
            if not ball_prim.IsValid():
                raise RuntimeError(f"红球 prim 无效: {self.prim_path}")

            ball_position, ball_quat = get_world_pose(self.prim_path)
            ball_position = np.asarray(ball_position, dtype=np.float64)
            candidates = []
            for body_path in body_paths if isinstance(body_paths, (list, tuple)) else [body_paths]:
                if body_path is None:
                    continue
                rigid_body_path = find_rigid_body_path(self.world.stage, body_path)
                if rigid_body_path is None:
                    continue
                body_prim = self.world.stage.GetPrimAtPath(rigid_body_path)
                if not body_prim.IsValid():
                    continue
                body_position, body_quat = get_world_pose(rigid_body_path)
                body_position = np.asarray(body_position, dtype=np.float64)
                distance = float(np.linalg.norm(ball_position - body_position))
                candidates.append((distance, rigid_body_path, body_position, body_quat))
            if not candidates:
                raise RuntimeError(f"没有从夹爪候选路径找到刚体 API: {body_paths}")

            body_origin_distance, rigid_body_path, body_position, body_quat = min(candidates, key=lambda item: item[0])
            self.last_attach_body_origin_distance = body_origin_distance
            if expected_ball_position is None:
                attach_distance = 0.0
            else:
                expected_ball_position = np.asarray(expected_ball_position, dtype=np.float64)
                if expected_ball_position.shape != (3,):
                    raise ValueError("物理抓取校验位置必须是长度为 3 的 xyz 数组")
                attach_distance = float(np.linalg.norm(ball_position - expected_ball_position))
            self.last_attach_distance = attach_distance
            if attach_distance > GRASP_ATTACH_MAX_DISTANCE_M:
                raise RuntimeError(
                    f"红球偏离视觉抓取点过远，不创建固定关节: "
                    f"distance={attach_distance:.3f}m limit={GRASP_ATTACH_MAX_DISTANCE_M:.3f}m "
                    f"body_origin_distance={body_origin_distance:.3f}m"
                )
            body_rotation = quat_to_rot_matrix(body_quat)
            local_pos0 = body_rotation.T @ (ball_position - body_position)
            local_rot0 = quat_multiply(quat_inverse(body_quat), ball_quat)

            self._remove_grasp_joint_locked()
            joint = UsdPhysics.FixedJoint.Define(self.world.stage, self.grasp_joint_path)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(rigid_body_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(self.prim_path)])
            joint.CreateBreakForceAttr().Set(float("inf"))
            joint.CreateBreakTorqueAttr().Set(float("inf"))
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2])))
            joint.CreateLocalRot0Attr().Set(
                Gf.Quatf(float(local_rot0[0]), float(local_rot0[1]), float(local_rot0[2]), float(local_rot0[3]))
            )
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            self._stop_ball_locked()
            force_reload_physics_from_usd()
            self.attached = True
            self.attached_body_path = rigid_body_path
            if GRASP_ASSISTED_ATTACHMENT_ENABLED:
                self.assisted_local_pos = local_pos0.copy()
                self.assisted_local_rot = local_rot0.copy()
                self.attachment_mode = "fixed_joint_assisted_pose"
            else:
                self.assisted_local_pos = None
                self.assisted_local_rot = None
                self.attachment_mode = "fixed_joint"
            return self.snapshot_locked()

    def attach_to_body_assisted(self, body_path, expected_ball_position=None, max_distance=None, mode="demo_assisted_pose"):
        with self.lock:
            ball_prim = self.world.stage.GetPrimAtPath(self.prim_path)
            body_prim = self.world.stage.GetPrimAtPath(body_path)
            if not ball_prim.IsValid():
                raise RuntimeError(f"红球 prim 无效: {self.prim_path}")
            if not body_prim.IsValid():
                raise RuntimeError(f"附着目标 prim 无效: {body_path}")

            ball_position, ball_quat = get_world_pose(self.prim_path)
            body_position, body_quat = get_world_pose(body_path)
            ball_position = np.asarray(ball_position, dtype=np.float64)
            body_position = np.asarray(body_position, dtype=np.float64)

            attach_distance = 0.0
            if expected_ball_position is not None:
                expected_ball_position = np.asarray(expected_ball_position, dtype=np.float64)
                if expected_ball_position.shape != (3,):
                    raise ValueError("demo 抓取校验位置必须是长度为 3 的 xyz 数组")
                attach_distance = float(np.linalg.norm(ball_position - expected_ball_position))
                if max_distance is not None and attach_distance > float(max_distance):
                    print(
                        f"[WARN] demo 红球物理位置偏离固定抓取点，强制同步: "
                        f"distance={attach_distance:.3f}m limit={float(max_distance):.3f}m"
                    )
                ball_position = expected_ball_position.copy()
                self.ball.set_world_pose(position=ball_position, orientation=ball_quat)
                self._stop_ball_locked()
                attach_distance = 0.0

            body_rotation = quat_to_rot_matrix(body_quat)
            local_pos = body_rotation.T @ (ball_position - body_position)
            local_rot = quat_multiply(quat_inverse(body_quat), ball_quat)
            self._remove_grasp_joint_locked()
            self._stop_ball_locked()
            self.attached = True
            self.attached_body_path = body_path
            self.assisted_local_pos = local_pos.copy()
            self.assisted_local_rot = local_rot.copy()
            self.attachment_mode = mode
            self.last_attach_distance = attach_distance
            self.last_attach_body_origin_distance = float(np.linalg.norm(ball_position - body_position))
            return self.snapshot_locked()

    def update_attached_pose(self):
        with self.lock:
            if not self.attached or self.attached_body_path is None or self.assisted_local_pos is None:
                return
            if not GRASP_ASSISTED_ATTACHMENT_ENABLED and self.attachment_mode != "demo_assisted_pose":
                return
            body_path = self.attached_body_path
            local_pos = self.assisted_local_pos.copy()
            local_rot = None if self.assisted_local_rot is None else self.assisted_local_rot.copy()
        try:
            body_position, body_quat = get_world_pose(body_path)
            body_position = np.asarray(body_position, dtype=np.float64)
            body_rotation = quat_to_rot_matrix(body_quat)
            target_position = body_position + body_rotation @ local_pos
            target_quat = quat_multiply(body_quat, local_rot) if local_rot is not None else None
            self.ball.set_world_pose(position=target_position, orientation=target_quat)
            self._stop_ball_locked()
        except Exception as exc:
            print(f"Warning: failed to update assisted red ball attachment: {exc}")

    def snapshot(self):
        with self.lock:
            return self.snapshot_locked()

    def snapshot_locked(self):
        prim = self.world.stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return {
                "prim_path": self.prim_path,
                "prim_valid": False,
                "position_xyz_m": None,
                "geometry": self.geometry,
                "size_xyz_m": self.object_size.tolist(),
                "radius_m": RED_BALL_RADIUS_M,
                "mass_kg": RED_BALL_MASS_KG,
                "disable_gravity": RED_BALL_DISABLE_GRAVITY,
                "attached": self.attached,
                "attached_body_path": self.attached_body_path,
                "attachment_mode": self.attachment_mode,
                "last_attach_distance_m": self.last_attach_distance,
                "last_attach_body_origin_distance_m": self.last_attach_body_origin_distance,
                "reset_pending": self.pending_reset is not None,
            }
        position, _ = get_world_pose(self.prim_path)
        has_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        return {
            "prim_path": self.prim_path,
            "prim_valid": True,
            "physical_entity": bool(has_rigid_body and has_collision and RED_BALL_MASS_KG > 0.0),
            "has_rigid_body_api": bool(has_rigid_body),
            "has_collision_api": bool(has_collision),
            "position_xyz_m": np.asarray(position, dtype=np.float64).tolist(),
            "geometry": self.geometry,
            "size_xyz_m": self.object_size.tolist(),
            "radius_m": RED_BALL_RADIUS_M,
            "mass_kg": RED_BALL_MASS_KG,
            "disable_gravity": RED_BALL_DISABLE_GRAVITY,
            "attached": self.attached,
            "attached_body_path": self.attached_body_path,
            "attachment_mode": self.attachment_mode,
            "last_attach_distance_m": self.last_attach_distance,
            "last_attach_body_origin_distance_m": self.last_attach_body_origin_distance,
            "reset_pending": self.pending_reset is not None,
        }


def make_static_scene(world):
    create_prim(
        "/World/TeleopLight",
        "DistantLight",
        attributes={"inputs:intensity": 3500, "inputs:angle": 0.8},
    )
    dome = UsdLux.DomeLight.Define(world.stage, "/World/TeleopDomeLight")
    dome.CreateIntensityAttr(420)
    ground = create_prim(
        "/World/TeleopGround",
        "Cube",
        position=np.array([0.3, 0.0, -0.015]),
        scale=np.array([1.0, 1.0, 0.02]),
        attributes={"primvars:displayColor": [(0.46, 0.49, 0.53)]},
    )
    UsdPhysics.CollisionAPI.Apply(ground)
    if not RED_BALL_ENABLED:
        return None
    if FIXED_DEMO_SUPPORT_ENABLED:
        support_size = np.asarray(FIXED_DEMO_SUPPORT_SIZE_M, dtype=np.float64)
        object_size = np.asarray(RED_OBJECT_SIZE_M, dtype=np.float64)
        top_z = float(FIXED_DEMO_BALL_POSITION[2] - 0.5 * object_size[2])
        support_height = max(0.02, min(float(support_size[2]), top_z))
        support_position = np.array(
            [FIXED_DEMO_BALL_POSITION[0], FIXED_DEMO_BALL_POSITION[1], 0.5 * support_height],
            dtype=np.float64,
        )
        world.scene.add(
            FixedCuboid(
                prim_path=FIXED_DEMO_SUPPORT_PATH,
                name="z1_fixed_demo_support",
                position=support_position,
                scale=np.array([support_size[0], support_size[1], support_height], dtype=np.float64),
                color=np.array([0.34, 0.36, 0.38]),
            )
        )
    return RedBallManager(world)
