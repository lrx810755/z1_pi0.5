#!/usr/bin/env python3
"""Open the training scene and load Isaac Sim's bundled ROS 2 bridge."""

import argparse
import json
import math
import os
import random
import sys
import time

from layout_sweep import layout_sweep_blockers
from spatial_grid import grid_cell_bounds
from z1_paths import asset_root, isaac_sim_root, repository_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default=str(
            asset_root() / "generated" / "smolvla_red_ball_into_box.usd"
        ),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--external-camera-layout", choices=("legacy", "dual"), default="legacy")
    parser.add_argument("--eval-table-friction", type=float, default=0.0)
    parser.add_argument("--eval-arm-max-velocity", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument(
        "--depth-streams",
        action="store_true",
        help=(
            "Publish the wrist and global depth render products. Both were "
            "published with zero subscribers while every extra render product "
            "takes render time away from the 10 Hz RGB grid that training "
            "recording samples, so depth stays off unless a task needs it."
        ),
    )
    parser.add_argument(
        "--grasp-config",
        default=os.environ.get(
            "Z1_ISAAC_GRASP_CONFIG",
            str(
                repository_root()
                / "z1_isaac_bridge"
                / "config"
                / "grasp_physics.json"
            ),
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
if not math.isfinite(ARGS.eval_table_friction) or ARGS.eval_table_friction < 0:
    raise ValueError("--eval-table-friction must be finite and nonnegative")
if not math.isfinite(ARGS.eval_arm_max_velocity) or ARGS.eval_arm_max_velocity < 0:
    raise ValueError("--eval-arm-max-velocity must be finite and nonnegative")
with open(ARGS.grasp_config, "r", encoding="utf-8") as stream:
    GRASP_CONFIG = json.load(stream)
GRASP_ENVELOPE = GRASP_CONFIG["grasp_envelope"]
TABLE_CONFIG = GRASP_CONFIG["table"]
COLLISION_PROXY_CONFIG = GRASP_CONFIG["collision_proxy"]
CONTACT_CONFIG = GRASP_CONFIG["contact_report"]

# Use the Humble libraries bundled with Isaac Sim. Standard DDS messages cross
# the process boundary, so the ROS workspace never enters Isaac's Python 3.11.
ISAAC_SIM_ROOT = str(isaac_sim_root())
BUNDLED_ROS_LIB = os.path.join(
    ISAAC_SIM_ROOT,
    "exts",
    "isaacsim.ros2.bridge",
    "humble",
    "lib",
) + os.sep
os.environ["ROS_DISTRO"] = "humble"
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
if os.name == "posix":
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in (os.environ.get("LD_LIBRARY_PATH", ""), BUNDLED_ROS_LIB)
        if value
    )

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "renderer": "RaytracedLighting",
    }
)

import carb
import numpy as np
import omni.graph.core as og
import omni.usd
import usdrt.Sdf
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils import extensions
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.xforms import get_world_pose as get_prim_world_pose
from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    UsdGeom,
    UsdPhysics,
    UsdShade,
)


COLOR_CAMERA = "/World/Z1/root_joint/camera_color_optical_frame/ColorCamera"
VLA_COLOR_CAMERA = (
    "/World/Z1/root_joint/vla_camera_color_optical_frame/VLAColorCamera"
)
LEFT_WRIST_CAMERA_FRAME = "/World/Z1/root_joint/gripperStator/LeftWristCameraOpticalFrame"
LEFT_WRIST_CAMERA = f"{LEFT_WRIST_CAMERA_FRAME}/LeftWristCamera"
DEPTH_CAMERA = "/World/Z1/root_joint/camera_depth_optical_frame/DepthCamera"
WRIST_COLOR_WIDTH = 640
WRIST_COLOR_HEIGHT = 360
WRIST_COLOR_FX = 451.606170654297
WRIST_COLOR_FY = 451.370849609375
WRIST_COLOR_CX = 325.197723388672
WRIST_COLOR_CY = 188.640960693359
CAMERA_QOS_PROFILE = json.dumps(
    {
        "history": "keepLast",
        "depth": 1,
        "reliability": "bestEffort",
        "durability": "volatile",
        "deadline": 0.0,
        "lifespan": 0.0,
        "liveliness": "systemDefault",
        "leaseDuration": 0.0,
    }
)
GLOBAL_CAMERA_FRAME = "/World/GlobalCamera/global_camera_optical_frame"
GLOBAL_CAMERA = f"{GLOBAL_CAMERA_FRAME}/GlobalCamera"
GLOBAL_CAMERA_FRAME_ID = "global_camera_optical_frame"
GRIPPER_STATOR = "/World/Z1/root_joint/gripperStator"
VLA_CAMERA_BODY = "/World/Z1/root_joint/vla_camera_link"
GRIPPER_MOVER = "/World/Z1/root_joint/gripperMover"
GRIPPER_STATOR_PAD = f"{GRIPPER_STATOR}/SmolVLAContactPad"
GRIPPER_MOVER_PAD = f"{GRIPPER_MOVER}/SmolVLAContactPad"
GRIPPER_RETENTION_LIP = f"{GRIPPER_STATOR}/SmolVLARetentionLip"
GLOBAL_CAMERA_POSITION = (0.72, -0.72, 0.72)
# world <- global_camera_optical_frame, quaternion in (w, x, y, z).
# The optical +Z axis looks at the table center (0.32, 0.02, 0.12), +X is
# image-right, and +Y is image-down.
GLOBAL_CAMERA_ORIENTATION = (
    0.44389558219778247,
    -0.8618642404075918,
    -0.2180293171642894,
    0.11229407851178168,
)
GLOBAL_CAMERA_FX = 554.2562584220408
GLOBAL_CAMERA_FY = 554.2562584220408
GLOBAL_CAMERA_CX = 320.0
GLOBAL_CAMERA_CY = 240.0
CAMERA_HORIZONTAL_APERTURE_MM = 20.955
TRAINING_OBJECT_PRIMS = (
    "/World/Task/Objects/Cup",
    "/World/Task/Objects/Apple",
    "/World/Task/Objects/Peach",
    "/World/Task/Objects/Mouse",
    "/World/Task/Objects/BeverageBottle",
)
SAFE_HOME_BY_JOINT = {
    "joint1": 0.0,
    "joint2": 0.002,
    "joint3": -0.002,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "jointGripper": -0.90,
}
OBJECT_NAME_TO_PRIM = {
    "cup": "/World/Task/Objects/Cup",
    "apple": "/World/Task/Objects/Apple",
    "peach": "/World/Task/Objects/Peach",
    "mouse": "/World/Task/Objects/Mouse",
    "beverage_bottle": "/World/Task/Objects/BeverageBottle",
}
ACTIVE_OBJECT_COLLIDER_PATHS = {
    "cup": f"{OBJECT_NAME_TO_PRIM['cup']}/Collider",
    "apple": f"{OBJECT_NAME_TO_PRIM['apple']}/SmolVLACollisionProxy",
    "peach": f"{OBJECT_NAME_TO_PRIM['peach']}/SmolVLACollisionProxy",
    "mouse": f"{OBJECT_NAME_TO_PRIM['mouse']}/SmolVLACollisionProxy",
    "beverage_bottle": (
        f"{OBJECT_NAME_TO_PRIM['beverage_bottle']}/Collider"
    ),
}
ACTIVE_COLLIDER_BY_OBJECT_ROOT = {
    OBJECT_NAME_TO_PRIM[name]: collider_path
    for name, collider_path in ACTIVE_OBJECT_COLLIDER_PATHS.items()
}
OBJECT_CENTER_Z_OFFSETS = GRASP_CONFIG["object_center_z_offsets_m"]
RANDOMIZATION_CONFIG = GRASP_CONFIG["scene_randomization"]
OBJECT_FOOTPRINT_RADII = {
    name: float(radius)
    for name, radius in RANDOMIZATION_CONFIG[
        "object_footprint_radii_m"
    ].items()
}
WORK_SURFACE = "/World/Task/WorkSurface"


def quaternion_conjugate(q):
    return np.asarray((q[0], -q[1], -q[2], -q[3]), dtype=np.float32)


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float32,
    )


def rotate_vector(q, value):
    vector_q = np.asarray((0.0, value[0], value[1], value[2]))
    return quaternion_multiply(
        quaternion_multiply(q, vector_q), quaternion_conjugate(q)
    )[1:]


def quaternion_angle(a, b):
    """Shortest rotation angle between normalized wxyz quaternions."""
    dot = min(
        1.0,
        abs(float(np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))),
    )
    return 2.0 * math.acos(dot)


def inside_grasp_envelope(relative_position):
    local_x, local_y, local_z = (
        float(value) for value in relative_position
    )
    return (
        float(GRASP_ENVELOPE["x_min_m"])
        <= local_x
        <= float(GRASP_ENVELOPE["x_max_m"])
        and abs(local_y) <= float(GRASP_ENVELOPE["y_abs_max_m"])
        and float(GRASP_ENVELOPE["z_min_m"])
        <= local_z
        <= float(GRASP_ENVELOPE["z_max_m"])
    )


def configure_grasp_friction(stage):
    """Use low table drag with a high-friction gripper/object pair."""

    def define_material(
        path, static_friction, dynamic_friction, friction_combine_mode
    ):
        material = UsdShade.Material.Define(stage, path)
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(static_friction)
        physics.CreateDynamicFrictionAttr(dynamic_friction)
        physics.CreateRestitutionAttr(0.0)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(
            material.GetPrim()
        )
        physx_material.CreateFrictionCombineModeAttr().Set(
            friction_combine_mode
        )
        physx_material.CreateRestitutionCombineModeAttr().Set("min")
        return material

    object_material = define_material(
        "/World/PhysicsMaterials/SmolVLAObject", 0.20, 0.15, "min"
    )
    # Use the object's low coefficient against the table, but the gripper's
    # high coefficient at the pinch.  This remains a collider-only grasp:
    # no hidden pad or constraint is introduced.
    gripper_material = define_material(
        "/World/PhysicsMaterials/SmolVLAGripper", 6.00, 5.00, "max"
    )
    bound_paths = []
    for path in (*TRAINING_OBJECT_PRIMS, GRIPPER_STATOR, GRIPPER_MOVER):
        material = (
            object_material
            if path in TRAINING_OBJECT_PRIMS
            else gripper_material
        )
        root = stage.GetPrimAtPath(path)
        if not root.IsValid():
            raise RuntimeError(f"Missing grasp friction prim: {path}")
        stack = [root]
        while stack:
            prim = stack.pop()
            stack.extend(prim.GetChildren())
            if prim != root and not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            bound_paths.append(str(prim.GetPath()))
    carb.log_info(
        "Bound grasp friction directly to "
        f"{len(bound_paths)} roots/colliders"
    )


def configure_eval_table_friction(stage, coefficient):
    """Raise object/table friction for evaluation without changing grasps."""
    if coefficient == 0:
        return
    table = stage.GetPrimAtPath(WORK_SURFACE)
    if not table.IsValid() or not table.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"Missing table collider: {WORK_SURFACE}")
    material = UsdShade.Material.Define(
        stage, "/World/PhysicsMaterials/EvalHighFrictionTable"
    )
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(coefficient)
    physics.CreateDynamicFrictionAttr(coefficient * 0.8)
    physics.CreateRestitutionAttr(0.0)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr().Set("max")
    physx.CreateRestitutionCombineModeAttr().Set("min")
    UsdShade.MaterialBindingAPI.Apply(table).Bind(
        material, materialPurpose="physics"
    )
    carb.log_info(
        f"Evaluation table friction: static={coefficient}, "
        f"dynamic={coefficient * 0.8}"
    )


def remove_gripper_contact_helpers(stage):
    """Remove every non-rendered collider formerly used to bridge finger gaps."""
    removed = []
    for path in (
        GRIPPER_STATOR_PAD,
        GRIPPER_MOVER_PAD,
        GRIPPER_RETENTION_LIP,
    ):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
            removed.append(path)
    carb.log_info(
        "Invisible gripper contact helpers are forbidden; removed="
        f"{removed}"
    )


def define_ellipsoid_convex_proxy(
    stage,
    collider_path,
    radii_m,
    center_z_m,
    latitude_segments=8,
    longitude_segments=12,
    flat_base_height_m=0.0,
    vertical_shape_power=1.0,
):
    """Author a low-poly ellipsoid mesh with one convex-hull collider.

    A scanned fruit that is represented by a complete ellipsoid has a
    single-point support at its lowest pole.  Such a proxy cannot remain
    upright after becoming dynamic, even when its velocity is exactly zero.
    ``flat_base_height_m`` flattens only the lowest latitude ring onto the
    original bottom plane, providing a small, visually hidden support patch
    without changing the proxy's height or widest grasping cross-section.
    """
    if len(radii_m) != 3 or any(float(value) <= 0.0 for value in radii_m):
        raise ValueError(f"Invalid ellipsoid radii: {radii_m!r}")
    latitude_segments = max(4, int(latitude_segments))
    longitude_segments = max(8, int(longitude_segments))
    radius_x, radius_y, radius_z = (float(value) for value in radii_m)
    flat_base_height_m = float(flat_base_height_m)
    vertical_shape_power = float(vertical_shape_power)
    if not 0.0 <= flat_base_height_m < radius_z:
        raise ValueError(
            "flat_base_height_m must be in [0, radius_z), got "
            f"{flat_base_height_m}"
        )
    if not 0.35 <= vertical_shape_power <= 1.0:
        raise ValueError(
            "vertical_shape_power must be in [0.35, 1.0], got "
            f"{vertical_shape_power}"
        )

    points = [Gf.Vec3f(0.0, 0.0, radius_z)]
    for latitude in range(1, latitude_segments):
        theta = math.pi * latitude / latitude_segments
        sin_theta = math.sin(theta) ** vertical_shape_power
        raw_cos_theta = math.cos(theta)
        cos_theta = math.copysign(
            abs(raw_cos_theta) ** vertical_shape_power,
            raw_cos_theta,
        )
        point_z = radius_z * cos_theta
        if point_z < -radius_z + flat_base_height_m:
            point_z = -radius_z
        for longitude in range(longitude_segments):
            phi = 2.0 * math.pi * longitude / longitude_segments
            points.append(
                Gf.Vec3f(
                    radius_x * sin_theta * math.cos(phi),
                    radius_y * sin_theta * math.sin(phi),
                    point_z,
                )
            )
    bottom_index = len(points)
    points.append(Gf.Vec3f(0.0, 0.0, -radius_z))

    face_counts = []
    face_indices = []
    first_ring = 1
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        face_counts.append(3)
        face_indices.extend(
            (0, first_ring + longitude, first_ring + next_longitude)
        )
    for latitude in range(latitude_segments - 2):
        ring = first_ring + latitude * longitude_segments
        next_ring = ring + longitude_segments
        for longitude in range(longitude_segments):
            next_longitude = (longitude + 1) % longitude_segments
            face_counts.append(4)
            face_indices.extend(
                (
                    ring + longitude,
                    next_ring + longitude,
                    next_ring + next_longitude,
                    ring + next_longitude,
                )
            )
    last_ring = bottom_index - longitude_segments
    for longitude in range(longitude_segments):
        next_longitude = (longitude + 1) % longitude_segments
        face_counts.append(3)
        face_indices.extend(
            (
                bottom_index,
                last_ring + next_longitude,
                last_ring + longitude,
            )
        )

    mesh = UsdGeom.Mesh.Define(stage, collider_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(center_z_m)))
    mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("convexHull")
    return mesh.GetPrim()


def configure_fruit_collision_approximations(stage):
    """Keep scan meshes visual-only and author stable convex colliders."""
    for object_name in ("apple", "peach", "mouse"):
        root_path = OBJECT_NAME_TO_PRIM[object_name]
        root = stage.GetPrimAtPath(root_path)
        disabled = 0
        stack = [root]
        while stack:
            prim = stack.pop()
            stack.extend(prim.GetChildren())
            collision = UsdPhysics.CollisionAPI(prim)
            if collision:
                collision.GetCollisionEnabledAttr().Set(False)
                disabled += 1

        proxy = COLLISION_PROXY_CONFIG[object_name]
        shape = proxy.get("shape")
        collider_path = f"{root_path}/SmolVLACollisionProxy"
        if shape == "sphere":
            collider = UsdGeom.Sphere.Define(stage, collider_path)
            collider.CreateRadiusAttr(float(proxy["radius_m"]))
            collider.AddTranslateOp().Set(
                Gf.Vec3d(
                    0.0,
                    0.0,
                    float(OBJECT_CENTER_Z_OFFSETS[object_name]),
                )
            )
            collider.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            collision = UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            collision.CreateCollisionEnabledAttr(True)
            proxy_description = (
                f"{float(proxy['radius_m']):.3f} m sphere"
            )
        elif shape == "ellipsoid":
            radii_m = tuple(float(value) for value in proxy["radii_m"])
            define_ellipsoid_convex_proxy(
                stage,
                collider_path,
                radii_m,
                OBJECT_CENTER_Z_OFFSETS[object_name],
                proxy.get("latitude_segments", 8),
                proxy.get("longitude_segments", 12),
                proxy.get("flat_base_height_m", 0.0),
                proxy.get("vertical_shape_power", 1.0),
            )
            proxy_description = (
                "ellipsoid convex hull with radii "
                f"{tuple(round(value, 4) for value in radii_m)} m, "
                "flat base height "
                f"{float(proxy.get('flat_base_height_m', 0.0)):.4f} m, "
                "vertical shape power "
                f"{float(proxy.get('vertical_shape_power', 1.0)):.2f}"
            )
        else:
            raise RuntimeError(
                f"Unsupported collision proxy for {object_name}: {proxy!r}"
            )
        rigid_physics = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        if "mass_kg" in proxy:
            mass = UsdPhysics.MassAPI.Apply(root)
            mass.CreateMassAttr().Set(float(proxy["mass_kg"]))
        rigid_physics.CreateLinearDampingAttr().Set(
            float(COLLISION_PROXY_CONFIG["fruit_linear_damping"])
        )
        rigid_physics.CreateAngularDampingAttr().Set(
            float(COLLISION_PROXY_CONFIG["fruit_angular_damping"])
        )
        carb.log_info(
            f"{object_name} visual scan kept for rendering; authored "
            f"{proxy_description} collision proxy "
            f"and disabled {disabled} imported colliders; "
            f"linear/angular damping="
            f"{float(COLLISION_PROXY_CONFIG['fruit_linear_damping']):.2f}/"
            f"{float(COLLISION_PROXY_CONFIG['fruit_angular_damping']):.2f}, "
            f"mass={proxy.get('mass_kg', 'asset-default')} kg"
        )


def configure_grasp_contact_offsets(stage):
    """Prevent default contact skins from ejecting a correctly centred fruit."""
    contact_offset = float(COLLISION_PROXY_CONFIG["contact_offset_m"])
    rest_offset = float(COLLISION_PROXY_CONFIG["rest_offset_m"])
    if contact_offset <= rest_offset:
        raise RuntimeError(
            "grasp contact_offset_m must be greater than rest_offset_m"
        )
    configured = []
    collision_prims = []
    gripper_mesh_count = 0
    for link_path in (GRIPPER_STATOR, GRIPPER_MOVER):
        collision_root = stage.GetPrimAtPath(f"{link_path}/collisions")
        if not collision_root.IsValid():
            continue
        # URDF importer makes each collision subtree an instance.  Instance
        # proxy attributes cannot be overridden per link, so expand these two
        # small gripper subtrees before setting millimetre-scale contact skins.
        if collision_root.IsInstance():
            collision_root.SetInstanceable(False)
        stack = [collision_root]
        while stack:
            prim = stack.pop()
            if prim.IsInstance():
                prim.SetInstanceable(False)
            stack.extend(prim.GetChildren())
            if UsdPhysics.CollisionAPI(prim):
                approximation = str(
                    COLLISION_PROXY_CONFIG[
                        "gripper_mesh_approximation"
                    ]
                )
                # Isaac's URDF importer authors CollisionAPI on an Xform
                # wrapper that references the STL, not on a local Mesh prim.
                # MeshCollisionAPI is intentionally applied to that wrapper.
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.CreateApproximationAttr().Set(
                    approximation
                )
                gripper_mesh_count += 1
                collision_prims.append(prim)
    collision_paths = tuple(ACTIVE_OBJECT_COLLIDER_PATHS.values()) + (
        "/World/Task/WorkSurface",
        "/World/Task/Box/Bottom",
        "/World/Task/Box/WallLeft",
        "/World/Task/Box/WallRight",
        "/World/Task/Box/WallFront",
        "/World/Task/Box/WallBack",
    )
    for collision_path in collision_paths:
        prim = stage.GetPrimAtPath(collision_path)
        if not prim.IsValid() or not UsdPhysics.CollisionAPI(prim):
            carb.log_warn(
                f"Grasp contact-offset prim unavailable: {collision_path}"
            )
            continue
        collision_prims.append(prim)
    for prim in collision_prims:
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision.CreateContactOffsetAttr(contact_offset)
        physx_collision.CreateRestOffsetAttr(rest_offset)
        configured.append(str(prim.GetPath()))
    if len(configured) < 10:
        raise RuntimeError(
            "Could not configure all gripper/fruit collision contact offsets: "
            f"{configured}"
        )
    if gripper_mesh_count < 2:
        raise RuntimeError(
            "Could not replace both gripper convex hulls with "
            "convex decomposition; collision prims="
            f"{[(str(prim.GetPath()), prim.GetTypeName()) for prim in collision_prims]}"
        )
    carb.log_info(
        f"Configured grasp contact offsets on {len(configured)} colliders: "
        f"contact={contact_offset:.4f} m rest={rest_offset:.4f} m; "
        f"gripper convex decompositions={gripper_mesh_count}"
    )


def configure_grasp_contact_reports(stage):
    """Author zero-threshold reports on both fingers and every target body."""
    report_paths = (GRIPPER_STATOR, GRIPPER_MOVER, *TRAINING_OBJECT_PRIMS)
    configured = []
    for path in report_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Contact-report prim unavailable: {path}")
        report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report.CreateThresholdAttr().Set(
            float(CONTACT_CONFIG.get("report_threshold_n", 0.0))
        )
        configured.append(path)
    carb.log_info(
        "PhysX grasp contact reports enabled on: " + ", ".join(configured)
    )
DRIVE_GAINS = {
    # Isaac position drives generally use stiffnesses around 1e5.  Lower
    # importer defaults let gravity leave joints 3/4 several degrees away
    # from the ROS command even after the trajectory had finished.
    "joint1": (100000.0, 1000.0, 1000.0),
    "joint2": (200000.0, 1000.0, 2000.0),
    "joint3": (400000.0, 1200.0, 1600.0),
    "joint4": (300000.0, 1200.0, 2000.0),
    "joint5": (50000.0, 500.0, 600.0),
    "joint6": (80000.0, 800.0, 800.0),
    # Keep the position interface, but make contact substantially less rigid.
    # This is not force control: the ROS assistant additionally detects a
    # stalled moving finger and retargets the drive to the measured angle.
    "jointGripper": (
        float(GRASP_CONFIG["gripper"]["drive_stiffness"]),
        float(GRASP_CONFIG["gripper"]["drive_damping"]),
        float(GRASP_CONFIG["gripper"]["max_effort"]),
    ),
}


class SceneResetInterface:
    """Expose a physics-safe scene reset service from Isaac's main thread."""

    def __init__(self, robot, objects, gripper_path):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self._String = String
        self._JointState = JointState
        self._Trigger = Trigger
        self._robot = robot
        self._objects = objects
        self._gripper_path = gripper_path
        self._frozen_path = None
        self._frozen_collisions_enabled = False
        self._kinematic_path = None
        self._frozen_position = None
        self._frozen_orientation = None
        self._target_object = "apple"
        self._target_grid = None
        self._collision_reenable_path = None
        self._collision_reenable_steps = 0
        self._contact_report_available = True
        self._active_contact_pairs = {}
        self._left_contact_object = None
        self._right_contact_object = None
        self._left_contact_collider = None
        self._right_contact_collider = None
        self._left_contact_point_world = None
        self._right_contact_point_world = None
        self._left_contact_impulse_ns = 0.0
        self._right_contact_impulse_ns = 0.0
        self._left_finger_touching_table = False
        self._right_finger_touching_table = False
        self._left_finger_table_impulse_ns = 0.0
        self._right_finger_table_impulse_ns = 0.0
        self._bilateral_contact_frames = 0
        self._object_table_contacts = set()
        self._object_table_penetrations_m = {}
        self._last_contact_status_publish_time = 0.0
        self._contact_status_publish_period_s = 0.1
        self._pending_randomization = None
        self._contact_report_sub = (
            get_physx_simulation_interface().subscribe_contact_report_events(
                self._on_contact_report_event
            )
        )
        if self._contact_report_sub is None:
            self._contact_report_available = False
            carb.log_warn(
                "PhysX contact-report subscription failed; grasp validation "
                "will fall back to geometric table clearance and will not "
                "claim bilateral contact."
            )
        self._initial_object_poses = {
            path: rigid.get_world_pose() for path, rigid in objects.items()
        }
        # The mouse's scanned mesh and rounded collision proxy share an 86 mm
        # long axis. Rotate the complete rigid (visual and collider together)
        # into the reachable short-axis grasp used by the 75-degree approach.
        mouse_path = OBJECT_NAME_TO_PRIM["mouse"]
        mouse_position, mouse_orientation = self._initial_object_poses[
            mouse_path
        ]
        yaw_90_wxyz = np.asarray(
            (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)),
            dtype=np.float32,
        )
        self._initial_object_poses[mouse_path] = (
            mouse_position,
            quaternion_multiply(yaw_90_wxyz, mouse_orientation),
        )
        self._context = rclpy.context.Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            "z1_isaac_scene_reset", context=self._context,
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._status_pub = self._node.create_publisher(
            String, "/smolvla/scene_reset_status", qos
        )
        self._grasp_status_pub = self._node.create_publisher(
            String, "/smolvla/scene_grasp_status", qos
        )
        # Reset must replace the command cached by the OmniGraph ROS joint
        # subscriber. Teleporting the articulation alone is temporary: the
        # next graph tick otherwise reapplies the failed trajectory endpoint.
        self._joint_command_pub = self._node.create_publisher(
            JointState, "/isaac_joint_commands", 10
        )
        self._node.create_subscription(
            String,
            "/smolvla/scene_grasp_command",
            self._grasp_command_callback,
            10,
        )
        self._node.create_subscription(
            String,
            "/smolvla/target_object",
            self._target_callback,
            10,
        )
        self._node.create_subscription(
            String,
            "/smolvla/scene_randomization_command",
            self._randomization_command_callback,
            10,
        )
        self._node.create_service(
            Trigger, "/smolvla/reset_scene", self._reset_callback
        )
        self._node.create_service(
            Trigger, "/smolvla/randomize_scene", self._randomize_callback
        )
        self._node.create_service(
            Trigger,
            "/smolvla/validate_randomized_scene",
            self._validate_randomized_scene_callback,
        )
        self._node.create_service(
            Trigger,
            "/smolvla/randomize_scene_easy",
            self._randomize_easy_callback,
        )
        self._node.create_service(
            Trigger,
            "/smolvla/randomize_scene_medium",
            self._randomize_medium_callback,
        )
        self._node.create_service(
            Trigger,
            "/smolvla/randomize_scene_full",
            self._randomize_full_callback,
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._publish_status("READY", "Isaac 场景复位与随机化服务已就绪")

    @staticmethod
    def _contact_path(value):
        return str(PhysicsSchemaTools.intToSdfPath(value))

    @staticmethod
    def _path_is(path, root):
        return path == root or path.startswith(root + "/")

    def _on_contact_report_event(self, contact_headers, contact_data):
        """Track the active collider pairs; summarize once per physics step."""
        try:
            for header in contact_headers:
                actor0 = self._contact_path(header.actor0)
                actor1 = self._contact_path(header.actor1)
                collider0 = self._contact_path(header.collider0)
                collider1 = self._contact_path(header.collider1)
                pair_key = tuple(sorted((collider0, collider1)))
                if header.type == ContactEventType.CONTACT_LOST:
                    self._active_contact_pairs.pop(pair_key, None)
                    continue
                if header.type not in (
                    ContactEventType.CONTACT_FOUND,
                    ContactEventType.CONTACT_PERSIST,
                ):
                    continue
                impulse = 0.0
                minimum_separation = math.inf
                positions = []
                normals = []
                begin = int(header.contact_data_offset)
                end = begin + int(header.num_contact_data)
                for index in range(begin, end):
                    impulse += float(
                        np.linalg.norm(
                            np.asarray(contact_data[index].impulse, dtype=float)
                        )
                    )
                    minimum_separation = min(
                        minimum_separation,
                        float(contact_data[index].separation),
                    )
                    positions.append(
                        np.asarray(contact_data[index].position, dtype=float)
                    )
                    normals.append(
                        np.asarray(contact_data[index].normal, dtype=float)
                    )
                self._active_contact_pairs[pair_key] = {
                    "endpoint0": (actor0, collider0),
                    "endpoint1": (actor1, collider1),
                    "impulse_ns": impulse,
                    "minimum_separation_m": (
                        minimum_separation
                        if math.isfinite(minimum_separation)
                        else 0.0
                    ),
                    "contact_position_world_m": (
                        np.mean(positions, axis=0).tolist()
                        if positions
                        else None
                    ),
                    "contact_normal_world": (
                        np.mean(normals, axis=0).tolist()
                        if normals
                        else None
                    ),
                }
        except Exception as exc:
            self._contact_report_available = False
            self._active_contact_pairs.clear()
            carb.log_error(f"PhysX contact-report callback disabled: {exc}")

    def _contact_object_for_endpoint(self, endpoint):
        for name, root in OBJECT_NAME_TO_PRIM.items():
            if any(self._path_is(path, root) for path in endpoint):
                return name
        return None

    def _endpoint_is(self, endpoint, root):
        return any(self._path_is(path, root) for path in endpoint)

    def _endpoint_is_actual_finger_collision(self, endpoint, root):
        """Accept only imported finger collision meshes, never helper prims."""
        _actor_path, collider_path = endpoint
        return self._path_is(collider_path, f"{root}/collisions")

    def update_contact_state(self):
        """Convert active PhysX pairs into continuous bilateral/table state."""
        if not self._contact_report_available:
            self._publish_contact_state_if_due()
            return
        left = {}
        right = {}
        left_table_contact = False
        right_table_contact = False
        left_table_impulse = 0.0
        right_table_impulse = 0.0
        table_contacts = set()
        table_penetrations = {}
        for contact in self._active_contact_pairs.values():
            endpoint0 = contact["endpoint0"]
            endpoint1 = contact["endpoint1"]
            object0 = self._contact_object_for_endpoint(endpoint0)
            object1 = self._contact_object_for_endpoint(endpoint1)
            impulse = float(contact["impulse_ns"])
            penetration = max(
                0.0, -float(contact.get("minimum_separation_m", 0.0))
            )
            contact_position = contact.get("contact_position_world_m")
            if (
                self._endpoint_is_actual_finger_collision(
                    endpoint0, GRIPPER_STATOR
                )
                and object1
                and impulse > 0.0
            ):
                if impulse >= left.get(object1, (0.0, "", None))[0]:
                    left[object1] = (impulse, endpoint0[1], contact_position)
            if (
                self._endpoint_is_actual_finger_collision(
                    endpoint1, GRIPPER_STATOR
                )
                and object0
                and impulse > 0.0
            ):
                if impulse >= left.get(object0, (0.0, "", None))[0]:
                    left[object0] = (impulse, endpoint1[1], contact_position)
            if (
                self._endpoint_is_actual_finger_collision(
                    endpoint0, GRIPPER_MOVER
                )
                and object1
                and impulse > 0.0
            ):
                if impulse >= right.get(object1, (0.0, "", None))[0]:
                    right[object1] = (impulse, endpoint0[1], contact_position)
            if (
                self._endpoint_is_actual_finger_collision(
                    endpoint1, GRIPPER_MOVER
                )
                and object0
                and impulse > 0.0
            ):
                if impulse >= right.get(object0, (0.0, "", None))[0]:
                    right[object0] = (impulse, endpoint1[1], contact_position)
            if self._endpoint_is(endpoint0, WORK_SURFACE) and object1:
                table_contacts.add(object1)
                table_penetrations[object1] = max(
                    table_penetrations.get(object1, 0.0), penetration
                )
            if self._endpoint_is(endpoint1, WORK_SURFACE) and object0:
                table_contacts.add(object0)
                table_penetrations[object0] = max(
                    table_penetrations.get(object0, 0.0), penetration
                )
            if (
                self._endpoint_is(endpoint0, GRIPPER_STATOR)
                and self._endpoint_is(endpoint1, WORK_SURFACE)
            ) or (
                self._endpoint_is(endpoint1, GRIPPER_STATOR)
                and self._endpoint_is(endpoint0, WORK_SURFACE)
            ):
                left_table_contact = True
                left_table_impulse = max(left_table_impulse, impulse)
            if (
                self._endpoint_is(endpoint0, GRIPPER_MOVER)
                and self._endpoint_is(endpoint1, WORK_SURFACE)
            ) or (
                self._endpoint_is(endpoint1, GRIPPER_MOVER)
                and self._endpoint_is(endpoint0, WORK_SURFACE)
            ):
                right_table_contact = True
                right_table_impulse = max(right_table_impulse, impulse)

        self._left_contact_object = (
            max(left, key=lambda name: left[name][0]) if left else None
        )
        self._right_contact_object = (
            max(right, key=lambda name: right[name][0]) if right else None
        )
        self._left_contact_impulse_ns = (
            left[self._left_contact_object][0]
            if self._left_contact_object is not None
            else 0.0
        )
        self._right_contact_impulse_ns = (
            right[self._right_contact_object][0]
            if self._right_contact_object is not None
            else 0.0
        )
        self._left_contact_collider = (
            left[self._left_contact_object][1]
            if self._left_contact_object is not None
            else None
        )
        self._right_contact_collider = (
            right[self._right_contact_object][1]
            if self._right_contact_object is not None
            else None
        )
        self._left_contact_point_world = (
            left[self._left_contact_object][2]
            if self._left_contact_object is not None
            else None
        )
        self._right_contact_point_world = (
            right[self._right_contact_object][2]
            if self._right_contact_object is not None
            else None
        )
        self._left_finger_touching_table = left_table_contact
        self._right_finger_touching_table = right_table_contact
        self._left_finger_table_impulse_ns = left_table_impulse
        self._right_finger_table_impulse_ns = right_table_impulse
        self._object_table_contacts = table_contacts
        self._object_table_penetrations_m = table_penetrations
        bilateral = (
            self._left_contact_object is not None
            and self._left_contact_object == self._right_contact_object
        )
        if bilateral:
            self._bilateral_contact_frames += 1
        else:
            self._bilateral_contact_frames = 0
        self._publish_contact_state_if_due()

    def _publish_contact_state_if_due(self):
        """Keep safety consumers supplied with fresh contact telemetry."""
        now = time.monotonic()
        if (
            now - self._last_contact_status_publish_time
            < self._contact_status_publish_period_s
        ):
            return
        self._last_contact_status_publish_time = now
        self._publish_grasp_status(
            "CONTACT_STATE",
            "实时接触状态",
            target=self._target_object,
            bilateral_contact=(
                self._left_contact_object is not None
                and self._left_contact_object == self._right_contact_object
            ),
        )

    def _clear_contact_state(self):
        self._active_contact_pairs.clear()
        self._left_contact_object = None
        self._right_contact_object = None
        self._left_contact_collider = None
        self._right_contact_collider = None
        self._left_contact_point_world = None
        self._right_contact_point_world = None
        self._left_contact_impulse_ns = 0.0
        self._right_contact_impulse_ns = 0.0
        self._left_finger_touching_table = False
        self._right_finger_touching_table = False
        self._left_finger_table_impulse_ns = 0.0
        self._right_finger_table_impulse_ns = 0.0
        self._bilateral_contact_frames = 0
        self._object_table_contacts.clear()
        self._object_table_penetrations_m.clear()

    def _target_callback(self, message):
        target = str(message.data).strip().lower()
        if target in OBJECT_NAME_TO_PRIM:
            self._target_object = target

    def _target_sweep_violation(
        self, object_name, position, target_position, target_orientation
    ):
        if object_name == self._target_object:
            return None
        blockers = layout_sweep_blockers(
            self._target_object,
            target_position,
            target_orientation,
            {
                self._target_object: target_position,
                object_name: position,
            },
            RANDOMIZATION_CONFIG,
        )
        return blockers.get(object_name)

    def _sample_randomized_object_poses_once(self, rng, level="easy"):
        profiles = RANDOMIZATION_CONFIG["levels"]
        if level not in profiles:
            raise ValueError(f"未知课程随机化等级：{level}")
        profile = profiles[level]
        mode = str(profile["mode"])
        delta = float(profile.get("translation_delta_m", 0.0))
        minimum_translation = float(
            profile["minimum_translation_m"]
        )
        yaw_delta = math.radians(
            float(profile["yaw_delta_deg"])
        )
        workspace_x = tuple(RANDOMIZATION_CONFIG["workspace_x_m"])
        workspace_y = tuple(RANDOMIZATION_CONFIG["workspace_y_m"])
        target_workspaces = profile.get("target_workspaces_m", {})
        box_x = tuple(RANDOMIZATION_CONFIG["box_exclusion_x_m"])
        box_y = tuple(RANDOMIZATION_CONFIG["box_exclusion_y_m"])
        clearance = float(
            RANDOMIZATION_CONFIG["minimum_object_clearance_m"]
        )
        max_attempts = int(
            RANDOMIZATION_CONFIG["maximum_sampling_attempts"]
        )
        sampled = {}
        target_object = self._target_object
        other_names = [
            name for name in OBJECT_NAME_TO_PRIM
            if name != target_object
        ]
        # Place the largest remaining footprint first. In the old fixed
        # object order the 100 mm mouse was placed near-last, so smaller
        # objects could consume every legal position in its local window.
        other_names.sort(
            key=lambda name: OBJECT_FOOTPRINT_RADII[name], reverse=True
        )
        object_names = [target_object, *other_names]
        for name in object_names:
            path = OBJECT_NAME_TO_PRIM[name]
            initial_position, initial_orientation = (
                self._initial_object_poses[path]
            )
            sampling_x = workspace_x
            sampling_y = workspace_y
            if mode == "workspace" and name == target_object:
                target_workspace = target_workspaces.get(name, {})
                sampling_x = tuple(
                    target_workspace.get("x", workspace_x)
                )
                sampling_y = tuple(
                    target_workspace.get("y", workspace_y)
                )
                if self._target_grid is not None:
                    sampling_x, sampling_y = grid_cell_bounds(
                        workspace_x,
                        workspace_y,
                        profile["target_grid_shape"],
                        self._target_grid,
                    )
            for _ in range(max_attempts):
                position = np.asarray(
                    initial_position, dtype=np.float32
                ).copy()
                if mode == "local":
                    position[0] += rng.uniform(-delta, delta)
                    position[1] += rng.uniform(-delta, delta)
                elif mode == "workspace":
                    position[0] = rng.uniform(*sampling_x)
                    position[1] = rng.uniform(*sampling_y)
                else:
                    raise ValueError(
                        f"课程等级 {level} 使用未知采样模式：{mode}"
                    )
                displacement = math.hypot(
                    float(position[0] - initial_position[0]),
                    float(position[1] - initial_position[1]),
                )
                if (
                    displacement < minimum_translation
                    and not (
                        name == target_object
                        and self._target_grid is not None
                    )
                ):
                    continue
                radius_with_clearance = (
                    OBJECT_FOOTPRINT_RADII[name] + clearance
                )
                if not (
                    workspace_x[0] <= float(position[0]) <= workspace_x[1]
                    and workspace_y[0] <= float(position[1]) <= workspace_y[1]
                ):
                    continue
                if (
                    box_x[0] - radius_with_clearance
                    <= float(position[0])
                    <= box_x[1] + radius_with_clearance
                    and box_y[0] - radius_with_clearance
                    <= float(position[1])
                    <= box_y[1] + radius_with_clearance
                ):
                    continue
                if name != target_object:
                    target_position, target_orientation = sampled[
                        target_object
                    ]
                    if self._target_sweep_violation(
                        name,
                        position,
                        target_position,
                        target_orientation,
                    ) is not None:
                        continue
                if all(
                    math.hypot(
                        float(position[0] - other_position[0]),
                        float(position[1] - other_position[1]),
                    )
                    >= (
                        OBJECT_FOOTPRINT_RADII[name]
                        + OBJECT_FOOTPRINT_RADII[other_name]
                        + clearance
                    )
                    for other_name, (other_position, _orientation) in (
                        sampled.items()
                    )
                ):
                    break
            else:
                raise RuntimeError(
                    f"无法为 {name} 采样互不重叠的桌面位置"
                )
            yaw = rng.uniform(-yaw_delta, yaw_delta)
            yaw_rotation = np.asarray(
                (
                    math.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                ),
                dtype=np.float32,
            )
            orientation = quaternion_multiply(
                yaw_rotation,
                np.asarray(initial_orientation, dtype=np.float32),
            )
            sampled[name] = (position, orientation)
        target_position, target_orientation = sampled[target_object]
        blockers = layout_sweep_blockers(
            target_object,
            target_position,
            target_orientation,
            {
                name: position
                for name, (position, _orientation) in sampled.items()
            },
            RANDOMIZATION_CONFIG,
        )
        if blockers:
            details = ", ".join(
                f"{name}({reason})"
                for name, reason in blockers.items()
            )
            raise RuntimeError(
                "布局未通过目标实际接近轴扫掠验证：" + details
            )
        return sampled

    def _sample_randomized_object_poses(self, seed, level="easy"):
        rng = random.Random(int(seed))
        layout_attempts = max(
            1,
            int(
                RANDOMIZATION_CONFIG.get(
                    "maximum_layout_sampling_attempts", 20
                )
            ),
        )
        last_error = None
        for _layout_attempt in range(layout_attempts):
            try:
                return self._sample_randomized_object_poses_once(
                    rng, level=level
                )
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(
            "整组布局重采样仍失败："
            f"attempts={layout_attempts}；last_error={last_error}"
        )

    def _apply_scene_randomization(
        self, seed, level="easy", auto_validate=False, settle_duration_s=None,
        non_target_max_angular_speed_rad_s=None,
    ):
        self._clear_frozen_object(publish=False)
        self._clear_contact_state()
        sampled = self._sample_randomized_object_poses(seed, level=level)
        zero_velocity = np.zeros(3, dtype=np.float32)
        requested_positions = {}
        requested_orientations = {}
        displacements_mm = []
        for name, (position, orientation) in sampled.items():
            path = OBJECT_NAME_TO_PRIM[name]
            rigid = self._objects[path]
            self._set_object_collisions(path, True)
            self._set_object_gravity(path, disabled=False)
            self._set_object_kinematic(path, False)
            rigid.set_world_pose(
                position=position, orientation=orientation
            )
            rigid.set_linear_velocity(zero_velocity)
            rigid.set_angular_velocity(zero_velocity)
            requested_positions[name] = [
                round(float(position[0]), 6),
                round(float(position[1]), 6),
                round(float(position[2]), 6),
            ]
            requested_orientations[name] = [
                round(float(value), 7) for value in orientation
            ]
            initial_position, _initial_orientation = (
                self._initial_object_poses[path]
            )
            displacements_mm.append(
                1000.0
                * math.hypot(
                    float(position[0] - initial_position[0]),
                    float(position[1] - initial_position[1]),
                )
            )
        movement_summary = (
            f"物体平移={min(displacements_mm):.0f}"
            f"–{max(displacements_mm):.0f} mm"
        )
        message = (
            f"随机位姿已写入，正在等待物理稳定；level={level}；"
            f"seed={int(seed)}；{movement_summary}"
        )
        self._pending_randomization = {
            "seed": int(seed),
            "level": level,
            "target_grid": (
                list(self._target_grid)
                if self._target_grid is not None
                else None
            ),
            "requested_poses": {
                name: (
                    np.asarray(position, dtype=np.float32).copy(),
                    np.asarray(orientation, dtype=np.float32).copy(),
                )
                for name, (position, orientation) in sampled.items()
            },
            "requested_positions_m": requested_positions,
            "requested_orientations_xyzw": requested_orientations,
            "object_displacements_mm": {
                name: round(displacement, 1)
                for name, displacement in zip(
                    sampled.keys(), displacements_mm
                )
            },
            "movement_summary": movement_summary,
            "applied_monotonic_s": time.monotonic(),
            "auto_validate": bool(auto_validate),
            "settle_duration_s": settle_duration_s,
            "non_target_max_angular_speed_rad_s": non_target_max_angular_speed_rad_s,
        }
        self._publish_status(
            "SETTLING",
            message,
            ready=False,
            curriculum_level=level,
            target_grid=self._pending_randomization["target_grid"],
            sweep_clearance_target=self._target_object,
            target_sweep_clear=True,
            seed=int(seed),
            requested_object_positions_m=requested_positions,
            object_displacements_mm=(
                self._pending_randomization["object_displacements_mm"]
            ),
        )
        return message

    def _validate_randomized_scene_callback(self, _request, response):
        pending = self._pending_randomization
        if pending is None:
            response.success = False
            response.message = "没有等待验收的随机场景"
            return response

        validation = RANDOMIZATION_CONFIG["validation"]
        minimum_settle_duration = float(
            validation["minimum_settle_duration_s"]
        )
        elapsed = time.monotonic() - float(
            pending["applied_monotonic_s"]
        )
        if elapsed < minimum_settle_duration:
            response.success = False
            response.message = (
                "随机场景尚未达到最短稳定时间："
                f"{elapsed:.3f} < {minimum_settle_duration:.3f} s"
            )
            return response

        max_horizontal_drift = float(
            validation["max_horizontal_drift_m"]
        )
        max_vertical_drift = float(
            validation["max_vertical_drift_m"]
        )
        max_linear_speed = float(
            validation["max_linear_speed_m_s"]
        )
        max_angular_speed = float(
            validation["max_angular_speed_rad_s"]
        )
        max_orientation_drift = math.radians(
            float(validation["max_orientation_drift_deg"])
        )
        max_table_penetration = float(
            validation["max_table_penetration_m"]
        )
        require_table_contact = bool(
            validation.get("require_table_contact", True)
        )

        failures = []
        settled_positions = {}
        settled_orientations = {}
        metrics = {}
        for name, (
            requested_position,
            requested_orientation,
        ) in pending["requested_poses"].items():
            path = OBJECT_NAME_TO_PRIM[name]
            rigid = self._objects[path]
            actual_position, actual_orientation = rigid.get_world_pose()
            actual_position = np.asarray(actual_position, dtype=np.float32)
            actual_orientation = np.asarray(
                actual_orientation, dtype=np.float32
            )
            position_delta = actual_position - requested_position
            horizontal_drift = math.hypot(
                float(position_delta[0]), float(position_delta[1])
            )
            vertical_drift = abs(float(position_delta[2]))
            orientation_drift = quaternion_angle(
                actual_orientation, requested_orientation
            )
            linear_speed = float(
                np.linalg.norm(
                    np.asarray(
                        rigid.get_linear_velocity(), dtype=np.float32
                    )
                )
            )
            angular_speed = float(
                np.linalg.norm(
                    np.asarray(
                        rigid.get_angular_velocity(), dtype=np.float32
                    )
                )
            )
            touching_table = name in self._object_table_contacts
            table_penetration = float(
                self._object_table_penetrations_m.get(name, 0.0)
            )
            settled_positions[name] = [
                round(float(value), 6) for value in actual_position
            ]
            settled_orientations[name] = [
                round(float(value), 7) for value in actual_orientation
            ]
            metrics[name] = {
                "horizontal_drift_m": round(horizontal_drift, 6),
                "vertical_drift_m": round(vertical_drift, 6),
                "linear_speed_m_s": round(linear_speed, 6),
                "angular_speed_rad_s": round(angular_speed, 6),
                "orientation_drift_deg": round(
                    math.degrees(orientation_drift), 3
                ),
                "touching_table": touching_table,
                "table_penetration_m": round(table_penetration, 6),
            }
            reasons = []
            if horizontal_drift > max_horizontal_drift:
                reasons.append(
                    f"横向漂移 {horizontal_drift:.4f} m"
                    f">{max_horizontal_drift:.4f} m"
                )
            if vertical_drift > max_vertical_drift:
                reasons.append(
                    f"竖向漂移 {vertical_drift:.4f} m"
                    f">{max_vertical_drift:.4f} m"
                )
            if linear_speed > max_linear_speed:
                reasons.append(
                    f"线速度 {linear_speed:.4f} m/s"
                    f">{max_linear_speed:.4f} m/s"
                )
            angular_limit = max_angular_speed
            if name != self._target_object:
                angular_limit = float(
                    pending.get("non_target_max_angular_speed_rad_s")
                    or max_angular_speed
                )
            if angular_speed > angular_limit:
                reasons.append(
                    f"角速度 {angular_speed:.4f} rad/s"
                    f">{angular_limit:.4f} rad/s"
                )
            if orientation_drift > max_orientation_drift:
                reasons.append(
                    f"姿态漂移 {math.degrees(orientation_drift):.2f}°"
                    f">{math.degrees(max_orientation_drift):.2f}°"
                )
            if (
                require_table_contact
                and self._contact_report_available
                and not touching_table
            ):
                reasons.append("未检测到与桌面的真实碰撞接触")
            if table_penetration > max_table_penetration:
                reasons.append(
                    f"桌面穿透 {table_penetration:.4f} m"
                    f">{max_table_penetration:.4f} m"
                )
            if reasons:
                failures.append(f"{name}: " + "，".join(reasons))

        details = {
            "curriculum_level": pending["level"],
            "target_grid": pending.get("target_grid"),
            "sweep_clearance_target": self._target_object,
            "target_sweep_clear": True,
            "seed": pending["seed"],
            "requested_object_positions_m": pending[
                "requested_positions_m"
            ],
            "object_positions_m": settled_positions,
            "object_orientations_xyzw": settled_orientations,
            "object_displacements_mm": pending[
                "object_displacements_mm"
            ],
            "settle_metrics": metrics,
            "settle_duration_s": round(elapsed, 3),
        }
        self._pending_randomization = None
        if failures:
            message = (
                "随机场景物理验收失败；"
                + "；".join(failures)
            )
            self._publish_status("ERROR", message, **details)
            response.success = False
            response.message = message
            return response

        message = (
            f"场景已随机化并通过物理验收；"
            f"level={pending['level']}；seed={pending['seed']}；"
            f"{pending['movement_summary']}"
        )
        self._publish_status("RANDOMIZED", message, **details)
        response.success = True
        response.message = message
        return response

    def _randomize_level(self, response, level):
        try:
            seed = time.time_ns() & 0x7FFFFFFF
            response.message = self._apply_scene_randomization(
                seed, level=level
            )
            response.success = True
        except Exception as exc:
            response.success = False
            response.message = f"Isaac 场景随机化失败：{exc}"
            self._publish_status("ERROR", response.message)
        return response

    def _randomize_callback(self, _request, response):
        return self._randomize_level(response, "easy")

    def _randomize_easy_callback(self, _request, response):
        return self._randomize_level(response, "easy")

    def _randomize_medium_callback(self, _request, response):
        return self._randomize_level(response, "medium")

    def _randomize_full_callback(self, _request, response):
        return self._randomize_level(response, "full")

    def _randomization_command_callback(self, message):
        try:
            command = json.loads(message.data)
            if not isinstance(command, dict):
                raise ValueError("随机化命令必须是 JSON 对象")
            if "target_grid" in command:
                target_grid = command["target_grid"]
                if target_grid is None:
                    self._target_grid = None
                else:
                    shape = RANDOMIZATION_CONFIG["levels"]["full"][
                        "target_grid_shape"
                    ]
                    # This validates dimensions and bounds in one place.
                    grid_cell_bounds(
                        RANDOMIZATION_CONFIG["workspace_x_m"],
                        RANDOMIZATION_CONFIG["workspace_y_m"],
                        shape,
                        target_grid,
                    )
                    self._target_grid = (
                        int(target_grid[0]), int(target_grid[1])
                    )
                if "seed" not in command:
                    self._publish_status(
                        "GRID_CONFIGURED",
                        f"目标采样方格已设置为 {self._target_grid}",
                        target_grid=(
                            list(self._target_grid)
                            if self._target_grid is not None
                            else None
                        ),
                    )
                    return
            if "seed" not in command:
                raise ValueError("随机化命令必须包含整数 seed")
            level = str(command.get("level", "easy")).strip().lower()
            self._apply_scene_randomization(
                int(command["seed"]),
                level=level,
                auto_validate=True,
                settle_duration_s=float(command.get("settle_duration_s", 0)) or None,
                non_target_max_angular_speed_rad_s=float(
                    command.get("non_target_max_angular_speed_rad_s", 0)
                ) or None,
            )
        except Exception as exc:
            self._publish_status(
                "ERROR", f"Isaac 可复现场景随机化失败：{exc}"
            )

    def _reset_callback(self, _request, response):
        try:
            self._pending_randomization = None
            self._clear_frozen_object(publish=False)
            for object_name in OBJECT_NAME_TO_PRIM:
                self._set_target_gravity(object_name, False)
            self._clear_contact_state()
            home_positions = np.asarray(
                [SAFE_HOME_BY_JOINT[name] for name in self._robot.dof_names],
                dtype=np.float32,
            )
            self._robot.set_joint_positions(home_positions)
            self._robot.set_joint_velocities(
                np.zeros(len(self._robot.dof_names), dtype=np.float32)
            )
            # Keep both the articulation drive and the ROS/OmniGraph command
            # cache at Home. This prevents the pre-reset hold command from
            # pulling the arm back to the failed pose on the next graph tick.
            self._robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=home_positions)
            )
            home_command = self._JointState()
            home_command.header.stamp = self._node.get_clock().now().to_msg()
            home_command.name = list(self._robot.dof_names)
            home_command.position = [
                float(value) for value in home_positions
            ]
            for _ in range(3):
                self._joint_command_pub.publish(home_command)
            zero_velocity = np.zeros(3, dtype=np.float32)
            for path, rigid in self._objects.items():
                # A failed guided alignment may be reset while a delayed
                # collision-filter refresh is still pending.  Restore every
                # object's complete dynamic state explicitly before
                # teleporting it; relying only on the currently tracked
                # frozen path can leave an untracked object collisionless and
                # let it fall through the table after reset.
                position, orientation = self._initial_object_poses[path]
                is_target = (
                    path == OBJECT_NAME_TO_PRIM[self._target_object]
                )
                self._set_object_collisions(path, is_target)
                self._set_object_gravity(path, disabled=not is_target)
                if not is_target:
                    position = np.asarray(position, dtype=np.float32).copy()
                    position[2] = -1.0
                rigid.set_world_pose(position=position, orientation=orientation)
                if is_target:
                    rigid.set_linear_velocity(zero_velocity)
                    rigid.set_angular_velocity(zero_velocity)
                self._set_object_kinematic(path, not is_target)
            response.success = True
            response.message = (
                "机械臂和夹爪已复位，Home 控制目标已同步；"
                f"{self._target_object} 保持真实动态碰撞，"
                "其余物体已移出本轮抓取路径"
            )
            self._publish_status("RESET", response.message)
        except Exception as exc:
            response.success = False
            response.message = f"Isaac 场景复位失败：{exc}"
            self._publish_status("ERROR", response.message)
        return response

    def _grasp_command_callback(self, message):
        try:
            command = json.loads(message.data)
            kind = str(command.get("command", ""))
            if kind == "freeze":
                self._freeze_object(str(command.get("target", "")),
                                    keep_collisions=bool(command.get("keep_collisions", False)))
            elif kind == "unfreeze":
                self._unfreeze_object(publish=True)
            elif kind == "probe":
                self._probe_object(str(command.get("target", "")))
            elif kind in ("enable_retention", "disable_retention"):
                raise RuntimeError(
                    "真实网格夹取模式不存在隐藏止滑唇或接触垫"
                )
            elif kind in ("gravity_compensate", "gravity_restore"):
                target = str(command.get("target", ""))
                compensated = kind == "gravity_compensate"
                self._set_target_gravity(target, compensated)
                self._publish_grasp_status(
                    (
                        "GRAVITY_COMPENSATED"
                        if compensated
                        else "GRAVITY_RESTORED"
                    ),
                    (
                        "无附着试抬重力补偿已启用"
                        if compensated
                        else "目标重力已恢复"
                    ),
                    target=target,
                )
            else:
                raise ValueError(f"未知抓取场景命令：{kind!r}")
        except Exception as exc:
            self._publish_grasp_status("ERROR", str(exc))

    def _set_target_gravity(self, target, compensate):
        path = OBJECT_NAME_TO_PRIM.get(target)
        if path is None:
            raise ValueError(f"未知训练物体：{target!r}")
        self._set_object_gravity(path, disabled=compensate)

    @staticmethod
    def _set_object_gravity(path, disabled):
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid.CreateDisableGravityAttr().Set(bool(disabled))

    def _freeze_object(self, target, keep_collisions=False):
        path = OBJECT_NAME_TO_PRIM.get(target)
        if path is None:
            raise ValueError(f"未知训练物体：{target!r}")
        if self._frozen_path == path:
            if self._frozen_collisions_enabled != keep_collisions:
                self._set_object_collisions(path, keep_collisions)
                self._frozen_collisions_enabled = keep_collisions
            self._publish_grasp_status(
                "FROZEN", f"{target} 已固定在接近起点", target=target,
                target_collisions_enabled=self._frozen_collisions_enabled,
            )
            return
        if self._frozen_path is not None:
            self._clear_frozen_object(publish=False)
        self._collision_reenable_path = None
        self._collision_reenable_steps = 0
        self._clear_contact_state()
        rigid = self._objects[path]
        gripper_position, _ = self._gripper_pose()
        object_position, _ = rigid.get_world_pose()
        distance = float(np.linalg.norm(object_position - gripper_position))
        if distance > 0.80:
            raise RuntimeError(
                f"{target} 距夹爪 {distance:.3f} m，尚未到达预抓取范围"
            )
        # Policy evaluation keeps a solid fixed target. Guided demonstrations
        # may explicitly use the legacy collision-free insertion mode.
        self._set_object_collisions(path, keep_collisions)
        self._frozen_collisions_enabled = keep_collisions
        self._set_object_kinematic(path, True)
        self._kinematic_path = path
        self._frozen_path = path
        self._frozen_position, self._frozen_orientation = rigid.get_world_pose()
        self._publish_grasp_status(
            "FROZEN", f"{target} 已固定在接近起点", target=target,
            target_collisions_enabled=keep_collisions,
        )

    def _unfreeze_object(self, publish):
        if self._frozen_path is not None:
            rigid = self._objects[self._frozen_path]
            self.update_freeze_state()
            # Drop the kinematic contact manifold before making the body
            # dynamic.  Otherwise PhysX can convert cached separation forces
            # into a one-frame impulse that ejects a correctly centred fruit.
            # Gravity must remain disabled during the two collision-free
            # filter-refresh ticks; otherwise the fruit free-falls several
            # millimetres before its table collider is restored.
            self._set_object_gravity(self._frozen_path, disabled=True)
            self._set_object_collisions(self._frozen_path, False)
            if self._kinematic_path == self._frozen_path:
                self._set_object_kinematic(self._frozen_path, False)
                self._kinematic_path = None
            rigid.set_linear_velocity(np.zeros(3, dtype=np.float32))
            rigid.set_angular_velocity(np.zeros(3, dtype=np.float32))
            self._collision_reenable_path = self._frozen_path
            self._collision_reenable_steps = 2
            self._clear_contact_state()
        self._frozen_path = None
        self._frozen_position = None
        self._frozen_orientation = None
        if publish:
            self._publish_grasp_status(
                "UNFROZEN", "目标恢复动态物理，等待稳定后低速闭爪"
            )
            if self._collision_reenable_path is None:
                self._publish_grasp_status("PHYSICS_READY", "目标动态物理已就绪")

    def _clear_frozen_object(self, publish):
        active_path = (
            self._frozen_path
            or self._collision_reenable_path
        )
        if active_path is not None:
            rigid = self._objects[active_path]
            self.update_freeze_state()
            self._set_object_collisions(active_path, True)
            self._set_object_gravity(active_path, disabled=False)
            if self._kinematic_path == active_path:
                self._set_object_kinematic(active_path, False)
                self._kinematic_path = None
            rigid.set_linear_velocity(np.zeros(3, dtype=np.float32))
            rigid.set_angular_velocity(np.zeros(3, dtype=np.float32))
        self._frozen_path = None
        self._kinematic_path = None
        self._frozen_position = None
        self._frozen_orientation = None
        self._collision_reenable_path = None
        self._collision_reenable_steps = 0
        if publish:
            self._publish_grasp_status(
                "UNFROZEN", "目标已恢复动态物理"
            )

    def _probe_object(self, target):
        path = OBJECT_NAME_TO_PRIM.get(target)
        if path is None:
            raise ValueError(f"未知训练物体：{target!r}")
        gripper_position, gripper_orientation = self._gripper_pose()
        rigid = self._objects[path]
        object_position, object_orientation = rigid.get_world_pose()
        object_center = object_position + np.asarray(
            (0.0, 0.0, OBJECT_CENTER_Z_OFFSETS[target]), dtype=np.float32
        )
        local_position = rotate_vector(
            quaternion_conjugate(gripper_orientation),
            object_center - gripper_position,
        )
        def contact_point_object_local(point_world):
            if point_world is None:
                return None
            local = rotate_vector(
                quaternion_conjugate(object_orientation),
                np.asarray(point_world, dtype=np.float32) - object_position,
            )
            return [round(float(value), 6) for value in local]

        left_contact_point_object = contact_point_object_local(
            self._left_contact_point_world
        )
        right_contact_point_object = contact_point_object_local(
            self._right_contact_point_world
        )
        velocity_available = True
        try:
            linear_velocity = np.asarray(
                rigid.get_linear_velocity(), dtype=np.float32
            )
            angular_velocity = np.asarray(
                rigid.get_angular_velocity(), dtype=np.float32
            )
        except Exception:
            velocity_available = False
            linear_velocity = np.full(3, np.nan, dtype=np.float32)
            angular_velocity = np.full(3, np.nan, dtype=np.float32)
        initial_position = self._initial_object_poses[path][0]
        if self._contact_report_available:
            object_touching_table = target in self._object_table_contacts
            object_touching_table_estimated = False
        else:
            object_touching_table = (
                float(object_position[2])
                <= float(initial_position[2])
                + float(TABLE_CONFIG["clearance_m"])
            )
            object_touching_table_estimated = True
        required_frames = int(CONTACT_CONFIG["required_consecutive_frames"])
        bilateral_contact = (
            self._left_contact_object == target
            and self._right_contact_object == target
        )
        sustained_bilateral_contact = (
            bilateral_contact
            and self._bilateral_contact_frames >= required_frames
        )
        gravity_disabled = bool(
            PhysxSchema.PhysxRigidBodyAPI.Apply(
                omni.usd.get_context().get_stage().GetPrimAtPath(path)
            ).GetDisableGravityAttr().Get()
        )
        self._publish_grasp_status(
            "PROBED",
            f"{target} 相对夹爪位置已测量",
            target=target,
            local_position_m=[round(float(value), 6) for value in local_position],
            gripper_position_m=[
                round(float(value), 7) for value in gripper_position
            ],
            gripper_orientation_wxyz=[
                round(float(value), 8) for value in gripper_orientation
            ],
            physics_enabled=self._kinematic_path != path,
            gravity_enabled=not gravity_disabled,
            object_position_m=[
                round(float(value), 6) for value in object_position
            ],
            object_orientation_wxyz=[
                round(float(value), 7) for value in object_orientation
            ],
            linear_velocity_m_s=[
                round(float(value), 6) for value in linear_velocity
            ],
            angular_velocity_rad_s=[
                round(float(value), 6) for value in angular_velocity
            ],
            linear_speed_m_s=(
                round(float(np.linalg.norm(linear_velocity)), 6)
                if velocity_available
                else None
            ),
            angular_speed_rad_s=(
                round(float(np.linalg.norm(angular_velocity)), 6)
                if velocity_available
                else None
            ),
            velocity_available=velocity_available,
            object_touching_table=object_touching_table,
            object_touching_table_estimated=object_touching_table_estimated,
            left_finger_contact=self._left_contact_object == target,
            right_finger_contact=self._right_contact_object == target,
            left_finger_contact_object=self._left_contact_object,
            right_finger_contact_object=self._right_contact_object,
            left_finger_contact_collider=self._left_contact_collider,
            right_finger_contact_collider=self._right_contact_collider,
            left_finger_contact_point_object_m=left_contact_point_object,
            right_finger_contact_point_object_m=right_contact_point_object,
            left_finger_contact_impulse_ns=self._left_contact_impulse_ns,
            right_finger_contact_impulse_ns=self._right_contact_impulse_ns,
            left_finger_touching_table=self._left_finger_touching_table,
            right_finger_touching_table=self._right_finger_touching_table,
            left_finger_table_impulse_ns=self._left_finger_table_impulse_ns,
            right_finger_table_impulse_ns=self._right_finger_table_impulse_ns,
            bilateral_contact_consecutive_frames=(
                self._bilateral_contact_frames
            ),
            bilateral_contact_required_frames=required_frames,
            bilateral_contact=bilateral_contact,
            sustained_bilateral_contact=sustained_bilateral_contact,
            contact_report_available=self._contact_report_available,
        )

    @staticmethod
    def _set_object_collisions(path, enabled):
        stage = omni.usd.get_context().get_stage()
        active_collider_path = ACTIVE_COLLIDER_BY_OBJECT_ROOT.get(path)
        if active_collider_path is None:
            raise RuntimeError(f"{path} 没有登记有效碰撞体")
        stack = [stage.GetPrimAtPath(path)]
        while stack:
            prim = stack.pop()
            stack.extend(prim.GetChildren())
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                # Imported/scanned assets can retain obsolete CollisionAPI
                # prims.  They are disabled when the runtime proxy is
                # authored and must never be revived by randomization or by
                # the freeze/unfreeze sequence.
                collision_enabled = bool(
                    enabled
                    and str(prim.GetPath()) == active_collider_path
                )
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(
                    collision_enabled
                )

    @staticmethod
    def _set_object_kinematic(path, enabled):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(path)
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        if not rigid_body:
            raise RuntimeError(f"{path} 没有 RigidBodyAPI")
        rigid_body.GetKinematicEnabledAttr().Set(bool(enabled))

    def update_freeze_state(self):
        if self._collision_reenable_path is not None:
            self._collision_reenable_steps -= 1
            if self._collision_reenable_steps <= 0:
                self._set_object_collisions(
                    self._collision_reenable_path, True
                )
                self._set_object_gravity(
                    self._collision_reenable_path, disabled=False
                )
                rigid = self._objects[self._collision_reenable_path]
                rigid.set_linear_velocity(np.zeros(3, dtype=np.float32))
                rigid.set_angular_velocity(np.zeros(3, dtype=np.float32))
                self._collision_reenable_path = None
                self._collision_reenable_steps = 0
                self._publish_grasp_status("PHYSICS_READY", "目标重力和碰撞已恢复")
        # A frozen kinematic body remains at the recorded pose without
        # per-frame writes.  This method only completes the two-step collision
        # and gravity restoration after unfreezing.
        pending = self._pending_randomization
        if pending is not None and bool(pending.get("auto_validate", False)):
            minimum_duration = max(
                float(RANDOMIZATION_CONFIG["validation"]["minimum_settle_duration_s"]),
                float(pending.get("settle_duration_s") or 0),
            )
            if (
                time.monotonic()
                - float(pending["applied_monotonic_s"])
                >= minimum_duration
            ):
                response = self._Trigger.Response()
                self._validate_randomized_scene_callback(None, response)
                if not response.success:
                    carb.log_error(response.message)

    def _gripper_pose(self):
        position, orientation = get_prim_world_pose(
            self._gripper_path, fabric=True
        )
        return (
            np.asarray(position, dtype=np.float32),
            np.asarray(orientation, dtype=np.float32),
        )

    def _publish_grasp_status(self, phase, message, **details):
        separations = []
        for pair in self._active_contact_pairs.values():
            endpoint0, endpoint1 = pair["endpoint0"], pair["endpoint1"]
            for finger, other in ((endpoint0, endpoint1), (endpoint1, endpoint0)):
                if self._contact_object_for_endpoint(other) != self._target_object:
                    continue
                if any(self._endpoint_is_actual_finger_collision(finger, root)
                       for root in (GRIPPER_STATOR, GRIPPER_MOVER)):
                    separations.append(float(pair.get("minimum_separation_m", 0.0)))
        state = {
            "target_finger_min_separation_m": min(separations) if separations else None,
            "target_finger_max_penetration_m": max(0.0, -min(separations)) if separations else None,
            "target_finger_contact_pair_count": len(separations),
            "frozen": self._frozen_path is not None,
            "target_collisions_enabled": self._frozen_collisions_enabled if self._frozen_path else self._collision_reenable_path is None,
            "contact_report_available": self._contact_report_available,
            "left_finger_contact_object": self._left_contact_object,
            "right_finger_contact_object": self._right_contact_object,
            "left_finger_contact_collider": self._left_contact_collider,
            "right_finger_contact_collider": self._right_contact_collider,
            "left_finger_contact_impulse_ns": self._left_contact_impulse_ns,
            "right_finger_contact_impulse_ns": self._right_contact_impulse_ns,
            "left_finger_touching_table": self._left_finger_touching_table,
            "right_finger_touching_table": self._right_finger_touching_table,
            "left_finger_table_impulse_ns": self._left_finger_table_impulse_ns,
            "right_finger_table_impulse_ns": self._right_finger_table_impulse_ns,
            "bilateral_contact_consecutive_frames": (
                self._bilateral_contact_frames
            ),
        }
        payload = self._String()
        payload.data = json.dumps(
            {"phase": phase, "message": message, **state, **details},
            ensure_ascii=False,
        )
        self._grasp_status_pub.publish(payload)

    def _publish_status(self, phase, message, **details):
        payload = self._String()
        payload.data = json.dumps(
            {
                "phase": phase,
                "ready": phase != "ERROR",
                "message": message,
                **details,
            },
            ensure_ascii=False,
        )
        self._status_pub.publish(payload)

    def spin_once(self):
        self._executor.spin_once(timeout_sec=0.0)

    def close(self):
        self._contact_report_sub = None
        self._executor.shutdown()
        self._node.destroy_node()
        self._context.shutdown()


def find_articulation_root(stage):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
    raise RuntimeError("No UsdPhysics.ArticulationRootAPI found in the scene")


def validate_prim(stage, path, description):
    if not stage.GetPrimAtPath(path).IsValid():
        raise RuntimeError(f"{description} prim not found: {path}")


def ensure_global_camera(stage):
    """Author the fixed third-person RGB-D camera if the USD predates it."""
    existing = stage.GetPrimAtPath(GLOBAL_CAMERA)
    if existing.IsValid():
        if not existing.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Global camera path is not a Camera: {GLOBAL_CAMERA}")
        return

    UsdGeom.Xform.Define(stage, "/World/GlobalCamera")
    optical = UsdGeom.Xform.Define(stage, GLOBAL_CAMERA_FRAME)
    optical_xform = UsdGeom.Xformable(optical.GetPrim())
    optical_xform.AddTranslateOp().Set(Gf.Vec3d(*GLOBAL_CAMERA_POSITION))
    w, x, y, z = GLOBAL_CAMERA_ORIENTATION
    optical_xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    camera = UsdGeom.Camera.Define(stage, GLOBAL_CAMERA)
    focal_length = (
        GLOBAL_CAMERA_FX * CAMERA_HORIZONTAL_APERTURE_MM / 640.0
    )
    vertical_aperture = focal_length * 480.0 / GLOBAL_CAMERA_FY
    camera.CreateHorizontalApertureAttr(CAMERA_HORIZONTAL_APERTURE_MM)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureOffsetAttr(0.0)
    camera.CreateVerticalApertureOffsetAttr(0.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5.0))
    # ROS optical frame uses +Z forward/+Y down; an Isaac camera looks along
    # local -Z with +Y up.
    UsdGeom.Xformable(camera.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3d(180.0, 0.0, 0.0)
    )


def ensure_external_cameras(stage):
    if ARGS.external_camera_layout != "dual":
        return
    config_path = repository_root() / "z1_isaac_bridge" / "config" / "external_cameras.json"
    configs = json.loads(config_path.read_text())
    for name, cfg in configs.items():
        frame = GLOBAL_CAMERA_FRAME if name == "global" else "/World/GlobalRightCamera/global_right_camera_optical_frame"
        path = GLOBAL_CAMERA if name == "global" else frame + "/GlobalRightCamera"
        optical = UsdGeom.Xform.Define(stage, frame)
        xf = UsdGeom.Xformable(optical.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*cfg["position_m"]))
        x, y, z, w = cfg["orientation_xyzw"]
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateHorizontalApertureAttr(CAMERA_HORIZONTAL_APERTURE_MM)
        camera.CreateVerticalApertureAttr(CAMERA_HORIZONTAL_APERTURE_MM * cfg["height"] / cfg["width"])
        camera.CreateFocalLengthAttr(cfg["fx"] * CAMERA_HORIZONTAL_APERTURE_MM / cfg["width"])
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5.0))
        camera_xf = UsdGeom.Xformable(camera.GetPrim())
        camera_xf.ClearXformOpOrder()
        camera_xf.AddRotateXYZOp().Set(Gf.Vec3d(180., 0., 0.))


def ensure_left_wrist_camera(stage):
    """Mount a virtual RGB camera on the fixed side of the gripper."""
    validate_prim(stage, GRIPPER_STATOR, "gripper stator")
    existing = stage.GetPrimAtPath(LEFT_WRIST_CAMERA)
    if existing.IsValid() and not existing.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Left wrist camera path is not a Camera: {LEFT_WRIST_CAMERA}")

    optical = UsdGeom.Xform.Define(stage, LEFT_WRIST_CAMERA_FRAME)
    xform = UsdGeom.Xformable(optical.GetPrim())
    ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
    # Side of the open fingers, looking into the grasp gap.
    (ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()).Set(
        Gf.Vec3d(0.02, -0.08, 0.04)
    )
    orientation = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), -90.0)
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), -90.0)
        * Gf.Rotation(Gf.Vec3d(1, 0, 0), -5.0)
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), 15.0)
    )
    (ops.get(UsdGeom.XformOp.TypeOrient) or xform.AddOrientOp(
        UsdGeom.XformOp.PrecisionDouble
    )).Set(orientation.GetQuat())
    camera = UsdGeom.Camera.Define(stage, LEFT_WRIST_CAMERA)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 5.0))
    # Isaac's camera looks along -Z, whereas the optical frame uses +Z.
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_ops = {op.GetOpType(): op for op in camera_xform.GetOrderedXformOps()}
    (camera_ops.get(UsdGeom.XformOp.TypeRotateXYZ) or camera_xform.AddRotateXYZOp()).Set(
        Gf.Vec3d(180.0, 0.0, 0.0)
    )
    configure_color_camera(stage, LEFT_WRIST_CAMERA)
    # Wider view keeps the target visible beside an open finger.
    camera.GetFocalLengthAttr().Set(
        250.0 * CAMERA_HORIZONTAL_APERTURE_MM / float(WRIST_COLOR_WIDTH)
    )


def disable_vla_camera_body(stage):
    """Keep the VLA optical frame but remove its old housing and collider."""
    for child in ("visuals", "collisions"):
        prim = stage.GetPrimAtPath(f"{VLA_CAMERA_BODY}/{child}")
        if prim.IsValid():
            prim.SetActive(False)


def configure_color_camera(stage, camera_path):
    """Apply the D435i factory-calibrated 640x360 RGB projection."""
    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
    focal_length = (
        WRIST_COLOR_FX
        * CAMERA_HORIZONTAL_APERTURE_MM
        / float(WRIST_COLOR_WIDTH)
    )
    vertical_aperture = (
        focal_length * float(WRIST_COLOR_HEIGHT) / WRIST_COLOR_FY
    )
    camera.CreateHorizontalApertureAttr(CAMERA_HORIZONTAL_APERTURE_MM)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureOffsetAttr(
        (WRIST_COLOR_CX - WRIST_COLOR_WIDTH / 2.0) / WRIST_COLOR_FX
    )
    camera.CreateVerticalApertureOffsetAttr(
        (WRIST_COLOR_CY - WRIST_COLOR_HEIGHT / 2.0) / WRIST_COLOR_FY
    )


def configure_wrist_color_cameras(stage):
    configure_color_camera(stage, COLOR_CAMERA)
    configure_color_camera(stage, VLA_COLOR_CAMERA)


def configure_drive_gains(articulation_root):
    robot = SingleArticulation(articulation_root)
    robot.initialize()
    controller = robot.get_articulation_controller()
    kps, kds = controller.get_gains()
    max_efforts = controller.get_max_efforts()
    for index, name in enumerate(robot.dof_names):
        if name not in DRIVE_GAINS:
            continue
        kps[index], kds[index], max_efforts[index] = DRIVE_GAINS[name]
    controller.set_gains(kps=kps, kds=kds)
    controller.set_max_efforts(max_efforts)
    if ARGS.eval_arm_max_velocity:
        arm_names = [name for name in robot.dof_names if name in DRIVE_GAINS and name != "jointGripper"]
        view = robot._articulation_view
        target = np.full((1, len(arm_names)), ARGS.eval_arm_max_velocity, dtype=np.float32)
        view.set_max_joint_velocities(target, joint_names=arm_names)
        actual = view.get_joint_max_velocities(joint_names=arm_names)
        if not np.allclose(actual, target):
            raise RuntimeError(f"Could not set eval arm joint velocity limits: {actual}")
        print(f"Eval arm joint velocity limit: {ARGS.eval_arm_max_velocity:.3f} rad/s")
    print("Configured Z1 drive gains:")
    for index, name in enumerate(robot.dof_names):
        if name in DRIVE_GAINS:
            print(
                f"  {name}: kp={kps[index]:.1f} kd={kds[index]:.1f} "
                f"max_effort={max_efforts[index]:.1f}"
            )
    return robot


def initialize_training_objects():
    objects = {}
    for index, path in enumerate(TRAINING_OBJECT_PRIMS):
        rigid = SingleRigidPrim(path, name=f"training_object_{index}")
        rigid.initialize()
        objects[path] = rigid
    return objects


def create_joint_graph(articulation_root):
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_Z1", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("Tick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.SET_VALUES: [
                ("PublishJointState.inputs:topicName", "joint_states"),
                ("SubscribeJointState.inputs:topicName", "isaac_joint_commands"),
                (
                    "PublishJointState.inputs:targetPrim",
                    [usdrt.Sdf.Path(articulation_root)],
                ),
                (
                    "ArticulationController.inputs:targetPrim",
                    [usdrt.Sdf.Path(articulation_root)],
                ),
            ],
            keys.CONNECT: [
                ("Tick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("Tick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("Tick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("Tick.outputs:tick", "PublishClock.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
            ],
        },
    )


def create_object_truth_graph():
    """Publish dynamic object poses on a dedicated TFMessage topic."""
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_ObjectTruth", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("Tick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishObjectTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.SET_VALUES: [
                ("PublishObjectTF.inputs:topicName", "/isaac/object_transforms"),
                (
                    "PublishObjectTF.inputs:targetPrims",
                    [usdrt.Sdf.Path(path) for path in TRAINING_OBJECT_PRIMS],
                ),
            ],
            keys.CONNECT: [
                ("Tick.outputs:tick", "PublishObjectTF.inputs:execIn"),
                ("Context.outputs:context", "PublishObjectTF.inputs:context"),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishObjectTF.inputs:timeStamp",
                ),
            ],
        },
    )


def create_camera_nodes(
    graph_path,
    camera_path,
    frame_id,
    image_topic,
    info_topic,
    image_type,
    width=640,
    height=480,
    frame_skip_count=0,
):
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("Tick", "omni.graph.action.OnPlaybackTick"),
                ("RunOneSimulationFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("PublishImage", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("PublishCameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.SET_VALUES: [
                ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_path)]),
                ("CreateRenderProduct.inputs:height", height),
                ("CreateRenderProduct.inputs:width", width),
                ("PublishImage.inputs:topicName", image_topic.lstrip("/")),
                ("PublishImage.inputs:type", image_type),
                ("PublishImage.inputs:frameId", frame_id),
                ("PublishImage.inputs:frameSkipCount", frame_skip_count),
                ("PublishImage.inputs:queueSize", 1),
                ("PublishImage.inputs:qosProfile", CAMERA_QOS_PROFILE),
                ("PublishCameraInfo.inputs:topicName", info_topic.lstrip("/")),
                ("PublishCameraInfo.inputs:frameId", frame_id),
                # Keep image data and its calibration metadata on the same
                # publication cadence.  Each helper owns a separate
                # synthetic-data gate in Isaac Sim.
                ("PublishCameraInfo.inputs:frameSkipCount", frame_skip_count),
                ("PublishCameraInfo.inputs:queueSize", 1),
                ("PublishCameraInfo.inputs:qosProfile", CAMERA_QOS_PROFILE),
            ],
            keys.CONNECT: [
                ("Tick.outputs:tick", "RunOneSimulationFrame.inputs:execIn"),
                ("RunOneSimulationFrame.outputs:step", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "PublishImage.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "PublishCameraInfo.inputs:execIn"),
                (
                    "CreateRenderProduct.outputs:renderProductPath",
                    "PublishImage.inputs:renderProductPath",
                ),
                (
                    "CreateRenderProduct.outputs:renderProductPath",
                    "PublishCameraInfo.inputs:renderProductPath",
                ),
            ],
        },
    )


def main():
    if not os.path.isfile(ARGS.scene):
        raise FileNotFoundError(ARGS.scene)

    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()
    if not open_stage(ARGS.scene):
        raise RuntimeError(f"Failed to open stage: {ARGS.scene}")
    while is_stage_loading():
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    ensure_global_camera(stage)
    ensure_external_cameras(stage)
    disable_vla_camera_body(stage)
    ensure_left_wrist_camera(stage)
    articulation_root = find_articulation_root(stage)
    validate_prim(stage, COLOR_CAMERA, "color camera")
    validate_prim(stage, VLA_COLOR_CAMERA, "VLA wrist color camera")
    validate_prim(stage, DEPTH_CAMERA, "depth camera")
    configure_wrist_color_cameras(stage)
    for path in TRAINING_OBJECT_PRIMS:
        validate_prim(stage, path, "training object")
    configure_fruit_collision_approximations(stage)
    remove_gripper_contact_helpers(stage)
    configure_grasp_friction(stage)
    configure_eval_table_friction(stage, ARGS.eval_table_friction)
    configure_grasp_contact_offsets(stage)
    configure_grasp_contact_reports(stage)

    create_joint_graph(articulation_root)
    create_object_truth_graph()
    # The current dataset and policy use only left wrist + global RGB. Keep
    # their render products alone so unused wrist streams cannot compete for
    # Isaac's camera rendering resources.
    create_camera_nodes(
        "/ROS2_LeftWristCamera",
        LEFT_WRIST_CAMERA,
        "left_wrist_camera_optical_frame",
        "/left_wrist_camera/color/image_raw",
        "/left_wrist_camera/color/camera_info",
        "rgb",
        width=WRIST_COLOR_WIDTH,
        height=WRIST_COLOR_HEIGHT,
        frame_skip_count=0,
    )
    create_camera_nodes(
        "/ROS2_GlobalColorCamera",
        GLOBAL_CAMERA,
        GLOBAL_CAMERA_FRAME_ID,
        "/global_camera/color/image_raw",
        "/global_camera/color/camera_info",
        "rgb",
        width=640,
        height=480,
        frame_skip_count=0,
    )
    if ARGS.external_camera_layout == "dual":
        create_camera_nodes("/ROS2_GlobalRightColorCamera",
            "/World/GlobalRightCamera/global_right_camera_optical_frame/GlobalRightCamera",
            "global_right_camera_optical_frame", "/global_right_camera/color/image_raw",
            "/global_right_camera/color/camera_info", "rgb", width=640, height=480, frame_skip_count=0)
    if ARGS.depth_streams:
        create_camera_nodes(
            "/ROS2_GlobalDepthCamera",
            GLOBAL_CAMERA,
            GLOBAL_CAMERA_FRAME_ID,
            "/global_camera/depth/image_rect_raw",
            "/global_camera/depth/camera_info",
            "depth",
        )
    simulation_app.update()

    context = SimulationContext(stage_units_in_meters=1.0)
    context.initialize_physics()
    context.play()
    # Let the OmniGraph articulation controller create its physics view first;
    # configuring gains before this tick is overwritten during its lazy init.
    context.step(render=False)
    robot = configure_drive_gains(articulation_root)
    training_objects = initialize_training_objects()
    reset_interface = SceneResetInterface(
        robot, training_objects, GRIPPER_STATOR
    )
    start = time.monotonic()
    print("========== Z1 ROS 2 / Isaac Sim ==========")
    print(f"Scene: {ARGS.scene}")
    print(f"Articulation: {articulation_root}")
    print("Joint command: /isaac_joint_commands")
    print("Joint state: /joint_states")
    print("Clock: /clock")
    print("Object truth: /isaac/object_transforms")
    depth_note = "color+depth" if ARGS.depth_streams else "color only"
    print(f"Left wrist camera (color only): /left_wrist_camera/...")
    print(f"Global camera ({depth_note}): /global_camera/...")
    print("Episode scene reset: /smolvla/reset_scene")

    try:
        while simulation_app.is_running():
            reset_interface.spin_once()
            reset_interface.update_freeze_state()
            context.step(render=True)
            reset_interface.update_contact_state()
            if ARGS.max_seconds > 0.0 and time.monotonic() - start >= ARGS.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        reset_interface.close()
        context.stop()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        carb.log_error(str(exc))
        simulation_app.close()
        sys.exit(1)
