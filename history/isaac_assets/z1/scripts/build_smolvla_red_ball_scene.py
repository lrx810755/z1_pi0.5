#!/usr/bin/env python3
"""Build the Z1+D435 Isaac asset and a multi-object SmolVLA tabletop scene.

Run with Isaac Sim's Python:

    /home/lp531/isaac-sim/python.sh \
      /home/lp531/isaac_assets/z1/scripts/build_smolvla_red_ball_scene.py

The input must be a plain URDF (expand the repository Xacro first).  Fixed
joints are deliberately not merged so the RealSense frames remain available.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from isaacsim import SimulationApp
from render_occlusion import author_close_range_refraction


def parse_args() -> argparse.Namespace:
    root = Path("/home/lp531/isaac_assets/z1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=root / "generated/z1_with_d435.urdf")
    parser.add_argument(
        "--resolved-urdf",
        type=Path,
        default=root / "generated/z1_with_d435_resolved.urdf",
    )
    parser.add_argument(
        "--z1-description-root",
        type=Path,
        default=Path("/home/lp531/ros2_ws/src/z1_ros2/z1_description"),
    )
    parser.add_argument("--robot-usd", type=Path, default=root / "generated/z1_with_d435.usd")
    parser.add_argument(
        "--scene-usd",
        type=Path,
        default=root / "generated/smolvla_red_ball_into_box.usd",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=root / "generated/smolvla_red_ball_into_box.json",
    )
    return parser.parse_args()


ARGS = parse_args()
sys.argv = [sys.argv[0]]
print("[BUILD] Starting Isaac Sim application", flush=True)
APP = SimulationApp({"headless": True})
print("[BUILD] Isaac Sim application ready", flush=True)

import omni.kit.commands
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
COLOR_IMAGE_WIDTH = 640
COLOR_IMAGE_HEIGHT = 360
# Factory intrinsics read from the physically mounted D435i at 640x360.
# Color and depth are different optical systems and must not share one K.
COLOR_FX = 451.606170654297
COLOR_FY = 451.370849609375
COLOR_CX = 325.197723388672
COLOR_CY = 188.640960693359
DEPTH_FX = 387.84747314453125
DEPTH_FY = 387.84747314453125
DEPTH_CX = 319.447265625
DEPTH_CY = 238.202484130859
HORIZONTAL_APERTURE_MM = 20.955
NEAR_CLIP_M = 0.05
FAR_CLIP_M = 5.0
GLOBAL_CAMERA_FRAME = "/World/GlobalCamera/global_camera_optical_frame"
GLOBAL_CAMERA_PATH = f"{GLOBAL_CAMERA_FRAME}/GlobalCamera"
GLOBAL_CAMERA_POSITION = (0.72, -0.72, 0.72)
GLOBAL_CAMERA_LOOK_AT = (0.32, 0.02, 0.12)
# world <- optical frame, quaternion (w, x, y, z). The fixed pose points the
# camera at GLOBAL_CAMERA_LOOK_AT with ROS optical +X right and +Y down.
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
ROBOT_DRIVE_GAINS = {
    "joint1": (100000.0, 1000.0, 1000.0),
    "joint2": (100000.0, 1000.0, 2000.0),
    "joint3": (150000.0, 1200.0, 1600.0),
    "joint4": (150000.0, 1200.0, 1000.0),
    "joint5": (50000.0, 500.0, 600.0),
    "joint6": (20000.0, 200.0, 300.0),
    "jointGripper": (2000.0, 100.0, 80.0),
}

BOX_CENTER_XY_M = (0.32, 0.14)
BOX_OUTER_SIZE_M = 0.16
BOX_WALL_THICKNESS_M = 0.01
BOX_WALL_HEIGHT_M = 0.10
BOX_BOTTOM_THICKNESS_M = 0.01
SUCCESS_VOLUME_CLEARANCE_M = 0.025
GRIPPER_RECOMMENDED_MAX_WIDTH_M = 0.055

TRAINING_OBJECTS = {
    "cup": {
        "path": "/World/Task/Objects/Cup",
        "position_m": (0.62, -0.08, 0.0),
        "yaw_deg": -20.0,
        "mass_kg": 0.08,
        "semantic_label": "cup",
        "visual_asset": "../scanned_objects/usd/cup.usd",
        "collision": {"shape": "cylinder", "radius_m": 0.0252, "height_m": 0.061},
        "grasp": {"strategy": "side_pinch", "width_m": 0.0504, "height_m": 0.035},
    },
    "apple": {
        "path": "/World/Task/Objects/Apple",
        "position_m": (0.44, -0.18, 0.0),
        "yaw_deg": 12.0,
        "mass_kg": 0.10,
        "semantic_label": "apple",
        "visual_asset": "../scanned_objects/usd/apple.usd",
        "collision": {
            "shape": "ellipsoid",
            "radii_m": (0.0176, 0.0175, 0.0190),
        },
        "grasp": {"strategy": "equatorial_pinch", "width_m": 0.0352, "height_m": 0.0190},
    },
    "peach": {
        "path": "/World/Task/Objects/Peach",
        "position_m": (0.54, -0.16, 0.0),
        "yaw_deg": -8.0,
        "mass_kg": 0.10,
        "semantic_label": "peach",
        "visual_asset": "../scanned_objects/usd/peach.usd",
        "collision": {
            "shape": "ellipsoid",
            "radii_m": (0.01545, 0.01515, 0.01450),
        },
        "grasp": {"strategy": "equatorial_pinch", "width_m": 0.0309, "height_m": 0.0145},
    },
    "mouse": {
        "path": "/World/Task/Objects/Mouse",
        "position_m": (0.49, -0.02, 0.0),
        "yaw_deg": 18.0,
        "mass_kg": 0.07,
        "semantic_label": "computer_mouse",
        "visual_asset": "../scanned_objects/usd/mouse.usd",
        # The scanned shell is rounded and tapered.  A full bounding box
        # exposes artificial corners that stop the fixed finger before the
        # real mouse can enter the open jaws at a steep approach angle.
        "collision": {
            "shape": "ellipsoid",
            "radii_m": (0.043, 0.0245, 0.01415),
        },
        "grasp": {
            "strategy": "side_pinch_across_short_axis",
            "width_m": 0.0490,
            "height_m": 0.020,
        },
    },
    "beverage_bottle": {
        "path": "/World/Task/Objects/BeverageBottle",
        "position_m": (0.62, 0.08, 0.0),
        "yaw_deg": 5.0,
        "mass_kg": 0.18,
        "semantic_label": "beverage_bottle",
        "visual_asset": "../scanned_objects/usd/beverage_bottle.usd",
        "collision": {"shape": "cylinder", "radius_m": 0.0242, "height_m": 0.157},
        "grasp": {
            "strategy": "side_pinch_on_body",
            "width_m": 0.0484,
            "height_m": 0.075,
        },
    },
}


def find_prim_by_name(stage: Usd.Stage, name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == name]
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(f"Expected one prim named {name!r}, found {paths}")
    return matches[0]


def set_xform(
    prim: Usd.Prim,
    translation: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if rotation_xyz_deg is not None:
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_xyz_deg))


def create_preview_material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float],
    roughness: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_visual_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def create_physics_material(
    stage: Usd.Stage,
    path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static_friction)
    api.CreateDynamicFrictionAttr(dynamic_friction)
    api.CreateRestitutionAttr(restitution)
    return material


def bind_physics_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def add_static_cube(
    stage: Usd.Stage,
    path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    visual_material: UsdShade.Material,
    physics_material: UsdShade.Material,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*position))
    xformable.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(prim)
    bind_visual_material(prim, visual_material)
    bind_physics_material(prim, physics_material)
    return prim


def add_rigid_object_root(
    stage: Usd.Stage,
    path: str,
    position: tuple[float, float, float],
    mass_kg: float,
    semantic_label: str,
    yaw_deg: float = 0.0,
) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_xform(prim, position, (0.0, 0.0, yaw_deg))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass_kg)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(True)
    prim.CreateAttribute("smolvla:semanticLabel", Sdf.ValueTypeNames.String, custom=True).Set(
        semantic_label
    )
    prim.CreateAttribute("smolvla:trainingObject", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    return prim


def add_object_cube(
    stage: Usd.Stage,
    path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    material: UsdShade.Material,
    physics_material: UsdShade.Material | None = None,
    collision: bool = False,
    visible: bool = True,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    set_xform(prim, position)
    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*size))
    bind_visual_material(prim, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        if physics_material is not None:
            bind_physics_material(prim, physics_material)
    if not visible:
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return prim


def add_object_cylinder(
    stage: Usd.Stage,
    path: str,
    position: tuple[float, float, float],
    radius: float,
    height: float,
    material: UsdShade.Material,
    physics_material: UsdShade.Material | None = None,
    collision: bool = False,
    visible: bool = True,
) -> Usd.Prim:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    prim = cylinder.GetPrim()
    set_xform(prim, position)
    bind_visual_material(prim, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        if physics_material is not None:
            bind_physics_material(prim, physics_material)
    if not visible:
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return prim


def add_object_sphere(
    stage: Usd.Stage,
    path: str,
    position: tuple[float, float, float],
    radius: float,
    material: UsdShade.Material,
    physics_material: UsdShade.Material | None = None,
    collision: bool = False,
    scale: tuple[float, float, float] | None = None,
    visible: bool = True,
) -> Usd.Prim:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    prim = sphere.GetPrim()
    set_xform(prim, position)
    if scale is not None:
        UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*scale))
    bind_visual_material(prim, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        if physics_material is not None:
            bind_physics_material(prim, physics_material)
    if not visible:
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return prim


def add_training_objects(
    stage: Usd.Stage,
    collision_preview: UsdShade.Material,
    object_physics: UsdShade.Material,
) -> None:
    UsdGeom.Xform.Define(stage, "/World/Task/Objects")
    for name, config in TRAINING_OBJECTS.items():
        root = add_rigid_object_root(
            stage,
            config["path"],
            config["position_m"],
            config["mass_kg"],
            config["semantic_label"],
            config["yaw_deg"],
        )
        root.CreateAttribute("smolvla:visualAsset", Sdf.ValueTypeNames.Asset, custom=True).Set(
            Sdf.AssetPath(config["visual_asset"])
        )
        root.CreateAttribute("smolvla:visualQuality", Sdf.ValueTypeNames.String, custom=True).Set(
            "high_quality_textured_scan"
        )
        visual = UsdGeom.Xform.Define(stage, f"{config['path']}/VisualAsset").GetPrim()
        visual.GetReferences().AddReference(config["visual_asset"])

        collision = config["collision"]
        collider_path = f"{config['path']}/Collider"
        if collision["shape"] == "box":
            size = collision["size_m"]
            add_object_cube(
                stage,
                collider_path,
                (0.0, 0.0, size[2] / 2.0),
                size,
                collision_preview,
                object_physics,
                collision=True,
                visible=False,
            )
        elif collision["shape"] == "sphere":
            radius = collision["radius_m"]
            add_object_sphere(
                stage,
                collider_path,
                (0.0, 0.0, radius),
                radius,
                collision_preview,
                object_physics,
                collision=True,
                visible=False,
            )
        elif collision["shape"] == "ellipsoid":
            radii = collision["radii_m"]
            add_object_sphere(
                stage,
                collider_path,
                (0.0, 0.0, radii[2]),
                1.0,
                collision_preview,
                object_physics,
                collision=True,
                scale=tuple(radii),
                visible=False,
            )
        elif collision["shape"] == "cylinder":
            height = collision["height_m"]
            add_object_cylinder(
                stage,
                collider_path,
                (0.0, 0.0, height / 2.0),
                collision["radius_m"],
                height,
                collision_preview,
                object_physics,
                collision=True,
                visible=False,
            )
        else:
            raise ValueError(f"Unsupported collision shape for {name}: {collision['shape']}")


def add_camera_metadata(
    prim: Usd.Prim,
    role: str,
    ros_frame: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> None:
    prim.CreateAttribute("smolvla:role", Sdf.ValueTypeNames.String, custom=True).Set(role)
    prim.CreateAttribute("smolvla:rosFrame", Sdf.ValueTypeNames.String, custom=True).Set(ros_frame)
    prim.CreateAttribute("smolvla:imageWidth", Sdf.ValueTypeNames.Int, custom=True).Set(image_width)
    prim.CreateAttribute("smolvla:imageHeight", Sdf.ValueTypeNames.Int, custom=True).Set(image_height)
    prim.CreateAttribute("smolvla:fx", Sdf.ValueTypeNames.Double, custom=True).Set(fx)
    prim.CreateAttribute("smolvla:fy", Sdf.ValueTypeNames.Double, custom=True).Set(fy)
    prim.CreateAttribute("smolvla:cx", Sdf.ValueTypeNames.Double, custom=True).Set(cx)
    prim.CreateAttribute("smolvla:cy", Sdf.ValueTypeNames.Double, custom=True).Set(cy)
    horizontal_fov_rad = 2.0 * math.atan(image_width / (2.0 * fx))
    prim.CreateAttribute("smolvla:horizontalFovRad", Sdf.ValueTypeNames.Double, custom=True).Set(
        horizontal_fov_rad
    )


def add_isaac_camera(
    stage: Usd.Stage,
    optical_frame_name: str,
    camera_name: str,
    role: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> str:
    optical_frame = find_prim_by_name(stage, optical_frame_name)
    camera_path = optical_frame.GetPath().AppendChild(camera_name)
    camera = UsdGeom.Camera.Define(stage, camera_path)
    focal_length_mm = fx * HORIZONTAL_APERTURE_MM / image_width
    vertical_aperture_mm = focal_length_mm * image_height / fy
    camera.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE_MM)
    camera.CreateVerticalApertureAttr(vertical_aperture_mm)
    camera.CreateFocalLengthAttr(focal_length_mm)
    # Isaac Sim 5.1 records these offsets but its renderer currently keeps the
    # projection center at the image center. Preserve the measured values in
    # both standard USD attributes and the exact SmolVLA metadata.
    camera.CreateHorizontalApertureOffsetAttr((cx - image_width / 2.0) / fx)
    camera.CreateVerticalApertureOffsetAttr((cy - image_height / 2.0) / fy)
    camera.CreateClippingRangeAttr(Gf.Vec2f(NEAR_CLIP_M, FAR_CLIP_M))

    # ROS optical frame: +Z forward, +X right, +Y down.
    # Isaac camera: local -Z forward, +X right, +Y up.
    set_xform(camera.GetPrim(), (0.0, 0.0, 0.0), (180.0, 0.0, 0.0))
    add_camera_metadata(
        camera.GetPrim(),
        role=role,
        ros_frame=optical_frame_name,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_width=image_width,
        image_height=image_height,
    )
    return str(camera_path)


def add_global_camera(stage: Usd.Stage) -> str:
    """Add the fixed third-person camera used by the multi-view dataset."""
    UsdGeom.Xform.Define(stage, "/World/GlobalCamera")
    optical = UsdGeom.Xform.Define(stage, GLOBAL_CAMERA_FRAME)
    optical_xform = UsdGeom.Xformable(optical.GetPrim())
    optical_xform.AddTranslateOp().Set(Gf.Vec3d(*GLOBAL_CAMERA_POSITION))
    w, x, y, z = GLOBAL_CAMERA_ORIENTATION
    optical_xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    camera = UsdGeom.Camera.Define(stage, GLOBAL_CAMERA_PATH)
    focal_length = GLOBAL_CAMERA_FX * HORIZONTAL_APERTURE_MM / IMAGE_WIDTH
    vertical_aperture = focal_length * IMAGE_HEIGHT / GLOBAL_CAMERA_FY
    camera.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE_MM)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureOffsetAttr(0.0)
    camera.CreateVerticalApertureOffsetAttr(0.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(NEAR_CLIP_M, FAR_CLIP_M))
    set_xform(camera.GetPrim(), (0.0, 0.0, 0.0), (180.0, 0.0, 0.0))
    add_camera_metadata(
        camera.GetPrim(),
        role="observation_global_rgbd",
        ros_frame="global_camera_optical_frame",
        fx=GLOBAL_CAMERA_FX,
        fy=GLOBAL_CAMERA_FY,
        cx=GLOBAL_CAMERA_CX,
        cy=GLOBAL_CAMERA_CY,
    )
    return GLOBAL_CAMERA_PATH


def configure_robot_drives(stage: Usd.Stage) -> None:
    configured = set()
    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in ROBOT_DRIVE_GAINS:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        stiffness, damping, max_force = ROBOT_DRIVE_GAINS[name]
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(max_force)
        configured.add(name)
    missing = sorted(set(ROBOT_DRIVE_GAINS) - configured)
    if missing:
        raise RuntimeError(f"Could not configure imported joint drives: {missing}")


def resolve_package_mesh_uris(source_urdf: Path, resolved_urdf: Path, package_root: Path) -> Path:
    """Resolve ROS package mesh URIs because Isaac's headless importer does not."""
    if not source_urdf.exists():
        raise FileNotFoundError(source_urdf)
    if not package_root.exists():
        raise FileNotFoundError(package_root)
    text = source_urdf.read_text(encoding="utf-8")
    package_prefix = "package://z1_description/"
    replacement = f"{package_root.resolve().as_posix()}/"
    count = text.count(package_prefix)
    if count == 0:
        raise RuntimeError(f"No {package_prefix!r} mesh URIs found in {source_urdf}")
    resolved = text.replace(package_prefix, replacement)
    resolved_urdf.parent.mkdir(parents=True, exist_ok=True)
    resolved_urdf.write_text(resolved, encoding="utf-8")

    missing_meshes = []
    for line in resolved.splitlines():
        marker = 'mesh filename="'
        if marker not in line:
            continue
        mesh_path = line.split(marker, 1)[1].split('"', 1)[0]
        if not Path(mesh_path).exists():
            missing_meshes.append(mesh_path)
    if missing_meshes:
        raise RuntimeError(f"Resolved URDF refers to missing meshes: {missing_meshes}")
    print(f"[BUILD] Resolved {count} package mesh URIs -> {resolved_urdf}", flush=True)
    return resolved_urdf


def prune_empty_importer_visual_references(urdf_path: Path, robot_usd_path: Path) -> None:
    """Remove importer-created visual references for URDF links with no visual.

    Isaac Sim 5.1 authors references such as ``/visuals/camera_depth_frame`` for
    every link, even when the URDF link is an empty TF frame.  Those references
    are harmless but unresolved and flood the log when the asset is opened.
    """
    root = ET.parse(urdf_path).getroot()
    visual_links = {
        link.get("name")
        for link in root.findall("link")
        if link.get("name") and link.find("visual") is not None
    }
    all_links = {link.get("name") for link in root.findall("link") if link.get("name")}
    empty_visual_links = sorted(all_links - visual_links)
    base_usd_path = robot_usd_path.parent / "configuration" / f"{robot_usd_path.stem}_base.usd"
    base_stage = Usd.Stage.Open(str(base_usd_path))
    if base_stage is None:
        raise RuntimeError(f"Cannot open importer base USD: {base_usd_path}")
    removed = []
    for link_name in empty_visual_links:
        path = f"/z1_description/{link_name}/visuals"
        if base_stage.GetPrimAtPath(path).IsValid():
            base_stage.RemovePrim(path)
            removed.append(path)
    base_stage.GetRootLayer().Save()
    print(f"[BUILD] Pruned {len(removed)} empty visual references from importer output", flush=True)


def import_robot(urdf_path: Path, robot_usd_path: Path) -> tuple[str, str]:
    if not urdf_path.exists():
        raise FileNotFoundError(urdf_path)
    robot_usd_path.parent.mkdir(parents=True, exist_ok=True)
    if robot_usd_path.exists():
        robot_usd_path.unlink()
    configuration_dir = robot_usd_path.parent / "configuration"
    if configuration_dir.exists():
        shutil.rmtree(configuration_dir)

    print("[BUILD] Creating URDF import config", flush=True)
    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    print(f"[BUILD] URDF import config status: {status}", flush=True)
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    config.set_merge_fixed_joints(False)
    print("[BUILD] Configured merge_fixed_joints", flush=True)
    config.set_replace_cylinders_with_capsules(False)
    config.set_convex_decomp(False)
    config.set_import_inertia_tensor(True)
    config.set_fix_base(True)
    config.set_self_collision(False)
    print("[BUILD] Configured geometry and base", flush=True)
    config.set_density(0.0)
    config.set_distance_scale(1.0)
    config.set_default_drive_type(1)
    config.set_default_drive_strength(1000.0)
    config.set_default_position_drive_damping(100.0)
    print("[BUILD] Configured scale and drives", flush=True)
    config.set_make_default_prim(True)
    config.set_parse_mimic(True)
    config.set_create_physics_scene(True)
    config.set_collision_from_visuals(False)
    print("[BUILD] Finished URDF import config", flush=True)

    print(f"[BUILD] Calling URDF importer -> {robot_usd_path}", flush=True)
    status, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path.resolve()),
        import_config=config,
        dest_path=str(robot_usd_path.resolve()),
        get_articulation_root=True,
    )
    print(f"[BUILD] URDF importer returned status={status} path={imported_path}", flush=True)
    if not status:
        raise RuntimeError("URDFParseAndImportFile failed")

    prune_empty_importer_visual_references(urdf_path, robot_usd_path)
    stage = Usd.Stage.Open(str(robot_usd_path))
    if stage is None:
        raise RuntimeError(f"Cannot open imported USD: {robot_usd_path}")
    configure_robot_drives(stage)

    color_path = add_isaac_camera(
        stage,
        optical_frame_name="camera_color_optical_frame",
        camera_name="ColorCamera",
        role="observation_rgb",
        fx=COLOR_FX,
        fy=COLOR_FY,
        cx=COLOR_CX,
        cy=COLOR_CY,
        image_width=COLOR_IMAGE_WIDTH,
        image_height=COLOR_IMAGE_HEIGHT,
    )
    vla_color_path = add_isaac_camera(
        stage,
        optical_frame_name="vla_camera_color_optical_frame",
        camera_name="VLAColorCamera",
        role="observation_rgb_vla_wrist",
        fx=COLOR_FX,
        fy=COLOR_FY,
        cx=COLOR_CX,
        cy=COLOR_CY,
        image_width=COLOR_IMAGE_WIDTH,
        image_height=COLOR_IMAGE_HEIGHT,
    )
    depth_path = add_isaac_camera(
        stage,
        optical_frame_name="camera_depth_optical_frame",
        camera_name="DepthCamera",
        role="observation_depth",
        fx=DEPTH_FX,
        fy=DEPTH_FY,
        cx=DEPTH_CX,
        cy=DEPTH_CY,
    )

    stage.GetRootLayer().customLayerData = {
        **stage.GetRootLayer().customLayerData,
        "sourceUrdf": str(urdf_path.resolve()),
        "cameraModel": "Intel RealSense D435",
        "cameraIntrinsicsSource": "Mounted Intel RealSense D435i factory calibration",
    }
    stage.GetRootLayer().Save()
    return color_path, depth_path, vla_color_path


def build_task_scene(
    robot_usd_path: Path,
    scene_usd_path: Path,
    color_camera_path: str,
    depth_camera_path: str,
    vla_color_camera_path: str,
) -> dict:
    scene_usd_path.parent.mkdir(parents=True, exist_ok=True)
    if scene_usd_path.exists():
        scene_usd_path.unlink()
    stage = Usd.Stage.CreateNew(str(scene_usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)

    robot = UsdGeom.Xform.Define(stage, "/World/Z1").GetPrim()
    relative_robot_asset = f"./{robot_usd_path.name}"
    robot.GetReferences().AddReference(relative_robot_asset)

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableStabilizationAttr(True)
    physx_scene.CreateSolverTypeAttr("TGS")

    UsdGeom.Scope.Define(stage, "/World/Looks")
    table_visual = create_preview_material(stage, "/World/Looks/Table", (0.58, 0.58, 0.58), 0.8)
    box_visual = create_preview_material(stage, "/World/Looks/Box", (0.12, 0.32, 0.72), 0.65)
    collision_preview = create_preview_material(
        stage, "/World/Looks/CollisionPreview", (0.35, 0.35, 0.35), 1.0
    )
    table_physics = create_physics_material(stage, "/World/Looks/TablePhysics", 0.8, 0.65, 0.02)
    box_physics = create_physics_material(stage, "/World/Looks/BoxPhysics", 0.65, 0.5, 0.05)
    object_physics = create_physics_material(stage, "/World/Looks/ObjectPhysics", 0.75, 0.6, 0.04)

    UsdGeom.Xform.Define(stage, "/World/Task")
    add_static_cube(
        stage,
        "/World/Task/WorkSurface",
        (0.30, 0.0, -0.025),
        (0.90, 0.70, 0.05),
        table_visual,
        table_physics,
    )

    add_training_objects(stage, collision_preview, object_physics)

    box_root = UsdGeom.Xform.Define(stage, "/World/Task/Box").GetPrim()
    box_root.CreateAttribute("smolvla:semanticLabel", Sdf.ValueTypeNames.String, custom=True).Set(
        "target_box"
    )
    bx, by = BOX_CENTER_XY_M
    outer = BOX_OUTER_SIZE_M
    wall = BOX_WALL_THICKNESS_M
    height = BOX_WALL_HEIGHT_M
    bottom = BOX_BOTTOM_THICKNESS_M
    wall_z = bottom + height / 2.0
    wall_offset = (outer - wall) / 2.0
    add_static_cube(
        stage,
        "/World/Task/Box/Bottom",
        (bx, by, bottom / 2.0),
        (outer, outer, bottom),
        box_visual,
        box_physics,
    )
    for name, position, size in (
        ("WallLeft", (bx - wall_offset, by, wall_z), (wall, outer, height)),
        ("WallRight", (bx + wall_offset, by, wall_z), (wall, outer, height)),
        ("WallFront", (bx, by - wall_offset, wall_z), (outer - 2.0 * wall, wall, height)),
        ("WallBack", (bx, by + wall_offset, wall_z), (outer - 2.0 * wall, wall, height)),
    ):
        add_static_cube(
            stage,
            f"/World/Task/Box/{name}",
            position,
            size,
            box_visual,
            box_physics,
        )

    goal = UsdGeom.Cube.Define(stage, "/World/Task/Box/SuccessVolume")
    goal.CreateSizeAttr(1.0)
    goal_xform = UsdGeom.Xformable(goal.GetPrim())
    goal_xform.AddTranslateOp().Set(Gf.Vec3d(bx, by, bottom + height / 2.0))
    goal_xform.AddScaleOp().Set(
        Gf.Vec3f(
            outer - 2.0 * wall - 2.0 * SUCCESS_VOLUME_CLEARANCE_M,
            outer - 2.0 * wall - 2.0 * SUCCESS_VOLUME_CLEARANCE_M,
            height,
        )
    )
    goal.GetPrim().CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
    goal.GetPrim().CreateAttribute("smolvla:semanticLabel", Sdf.ValueTypeNames.String, custom=True).Set(
        "success_volume"
    )

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(1800.0)
    key.CreateWidthAttr(1.0)
    key.CreateHeightAttr(1.0)
    set_xform(key.GetPrim(), (0.35, -0.25, 1.20), (0.0, 0.0, 0.0))
    global_camera_path = add_global_camera(stage)

    task_metadata = {
        "task": "multi_object_tabletop_manipulation",
        "language_instruction": "Manipulate the requested tabletop object.",
        "candidate_instructions": [
            "Put the cup into the blue box.",
            "Put the apple into the blue box.",
            "Put the peach into the blue box.",
            "Put the computer mouse into the blue box.",
            "Put the beverage bottle into the blue box.",
        ],
        "coordinate_frame": "Z1 link00; +X forward; +Z up",
        "robot_asset": str(robot_usd_path.resolve()),
        "scene_asset": str(scene_usd_path.resolve()),
        "camera": {
            "color_prim_in_robot_asset": color_camera_path,
            "depth_prim_in_robot_asset": depth_camera_path,
            "color_resolution": [COLOR_IMAGE_WIDTH, COLOR_IMAGE_HEIGHT],
            "depth_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "color_K": [
                [COLOR_FX, 0.0, COLOR_CX],
                [0.0, COLOR_FY, COLOR_CY],
                [0.0, 0.0, 1.0],
            ],
            "depth_K": [
                [DEPTH_FX, 0.0, DEPTH_CX],
                [0.0, DEPTH_FY, DEPTH_CY],
                [0.0, 0.0, 1.0],
            ],
            "color_horizontal_fov_rad": 2.0
            * math.atan(COLOR_IMAGE_WIDTH / (2.0 * COLOR_FX)),
            "depth_horizontal_fov_rad": 2.0
            * math.atan(IMAGE_WIDTH / (2.0 * DEPTH_FX)),
            "clipping_range_m": [NEAR_CLIP_M, FAR_CLIP_M],
        },
        "vla_wrist_camera": {
            "color_prim_in_robot_asset": vla_color_camera_path,
            "frame_id": "vla_camera_color_optical_frame",
            "resolution": [COLOR_IMAGE_WIDTH, COLOR_IMAGE_HEIGHT],
            "K": [
                [COLOR_FX, 0.0, COLOR_CX],
                [0.0, COLOR_FY, COLOR_CY],
                [0.0, 0.0, 1.0],
            ],
            "clipping_range_m": [NEAR_CLIP_M, FAR_CLIP_M],
            "camera_mount_xyz_m": [0.0243802760, -0.0325000000, 0.0900367127],
            "camera_mount_rpy_rad": [0.0, 0.5597969220, 0.0],
            "validation_source": "eight saved READY/GRASPING episode replays",
        },
        "global_camera": {
            "prim": global_camera_path,
            "frame_id": "global_camera_optical_frame",
            "resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "K": [
                [GLOBAL_CAMERA_FX, 0.0, GLOBAL_CAMERA_CX],
                [0.0, GLOBAL_CAMERA_FY, GLOBAL_CAMERA_CY],
                [0.0, 0.0, 1.0],
            ],
            "position_m": list(GLOBAL_CAMERA_POSITION),
            "orientation_wxyz": list(GLOBAL_CAMERA_ORIENTATION),
            "look_at_m": list(GLOBAL_CAMERA_LOOK_AT),
            "clipping_range_m": [NEAR_CLIP_M, FAR_CLIP_M],
        },
        "initial_state": {
            "box_center_xy_m": list(BOX_CENTER_XY_M),
            "objects": {
                name: {
                    "prim": config["path"],
                    "position_m": list(config["position_m"]),
                    "yaw_deg": config["yaw_deg"],
                    "mass_kg": config["mass_kg"],
                    "semantic_label": config["semantic_label"],
                    "visual_asset": config["visual_asset"],
                    "visual_quality": "high_quality_textured_scan",
                    "collision": config["collision"],
                    "grasp": {
                        **config["grasp"],
                        "compatible_with_z1_gripper": True,
                        "recommended_max_width_m": GRIPPER_RECOMMENDED_MAX_WIDTH_M,
                    },
                }
                for name, config in TRAINING_OBJECTS.items()
            },
        },
        "randomization_ranges": {
            "object_translation_delta_m": [-0.025, 0.025],
            "object_yaw_delta_deg": [-15.0, 15.0],
            "box_x_m": [0.28, 0.37],
            "box_y_m": [0.08, 0.18],
            "light_intensity_scale": [0.8, 1.2],
            "camera_translation_delta_m": [-0.005, 0.005],
            "camera_rotation_delta_deg": [-2.0, 2.0],
        },
        "success_condition": {
            "selected_object_center_inside": "/World/Task/Box/SuccessVolume",
            "maximum_object_speed_m_s": 0.03,
            "stable_duration_s": 0.5,
            "gripper_released": True,
        },
    }
    stage.GetRootLayer().customLayerData = {
        "taskName": task_metadata["task"],
        "languageInstruction": task_metadata["language_instruction"],
        "metersPerUnit": 1.0,
        "trainingObjectAssetManifest": "../scanned_objects/manifest.json",
        "trainingObjectVisualQuality": "high_quality_textured_scan",
    }
    author_close_range_refraction(stage)
    stage.GetRootLayer().Save()
    return task_metadata


def validate_outputs(robot_usd_path: Path, scene_usd_path: Path) -> dict:
    robot_stage = Usd.Stage.Open(str(robot_usd_path))
    scene_stage = Usd.Stage.Open(str(scene_usd_path))
    if robot_stage is None or scene_stage is None:
        raise RuntimeError("Failed to reopen generated USD files")

    required_robot_names = {
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "jointGripper",
        "gripperStator",
        "camera_link",
        "camera_color_optical_frame",
        "camera_depth_optical_frame",
        "vla_camera_link",
        "vla_camera_color_optical_frame",
        "ColorCamera",
        "DepthCamera",
        "VLAColorCamera",
    }
    found_names = {prim.GetName() for prim in robot_stage.Traverse()}
    missing = sorted(required_robot_names - found_names)
    if missing:
        raise RuntimeError(f"Generated robot USD is missing: {missing}")

    robot_prims_with_instances = list(
        Usd.PrimRange.Stage(robot_stage, Usd.TraverseInstanceProxies())
    )
    mesh_count = sum(1 for prim in robot_prims_with_instances if prim.IsA(UsdGeom.Mesh))
    collision_count = sum(
        1 for prim in robot_prims_with_instances if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    if mesh_count < 10:
        raise RuntimeError(f"Robot visual mesh composition is incomplete: mesh_count={mesh_count}")
    if collision_count < 10:
        raise RuntimeError(f"Robot collision composition is incomplete: collision_count={collision_count}")

    color = find_prim_by_name(robot_stage, "ColorCamera")
    camera = UsdGeom.Camera(color)
    expected_focal = COLOR_FX * HORIZONTAL_APERTURE_MM / IMAGE_WIDTH
    if not math.isclose(float(camera.GetFocalLengthAttr().Get()), expected_focal, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError("Color camera focal length does not match Gazebo fx")

    for path in (
        "/World/Z1",
        "/World/Task/WorkSurface",
        "/World/Task/Objects/Cup",
        "/World/Task/Objects/Apple",
        "/World/Task/Objects/Peach",
        "/World/Task/Objects/Mouse",
        "/World/Task/Objects/BeverageBottle",
        "/World/Task/Box/Bottom",
        "/World/Task/Box/WallLeft",
        "/World/Task/Box/WallRight",
        "/World/Task/Box/WallFront",
        "/World/Task/Box/WallBack",
        "/World/Task/Box/SuccessVolume",
        GLOBAL_CAMERA_FRAME,
        GLOBAL_CAMERA_PATH,
    ):
        if not scene_stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"Generated scene USD is missing {path}")

    training_object_mesh_count = 0
    for name, config in TRAINING_OBJECTS.items():
        if config["grasp"]["width_m"] > GRIPPER_RECOMMENDED_MAX_WIDTH_M:
            raise RuntimeError(
                f"{name} grasp width {config['grasp']['width_m']:.4f} m exceeds "
                f"the Z1 recommendation {GRIPPER_RECOMMENDED_MAX_WIDTH_M:.4f} m"
            )
        visual_path = f"{config['path']}/VisualAsset"
        collider_path = f"{config['path']}/Collider"
        visual = scene_stage.GetPrimAtPath(visual_path)
        collider = scene_stage.GetPrimAtPath(collider_path)
        if not visual.IsValid() or not visual.HasAuthoredReferences():
            raise RuntimeError(f"{name} is missing its high-quality visual reference at {visual_path}")
        if not collider.IsValid() or not collider.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"{name} is missing its lightweight collider at {collider_path}")
        training_object_mesh_count += sum(
            1
            for prim in Usd.PrimRange(visual, Usd.TraverseInstanceProxies())
            if prim.IsA(UsdGeom.Mesh)
        )
    if training_object_mesh_count < len(TRAINING_OBJECTS):
        raise RuntimeError(
            "High-quality object composition is incomplete: "
            f"mesh_count={training_object_mesh_count}"
        )
    if any(prim.GetName().lower() in {"redball", "red_ball"} for prim in scene_stage.Traverse()):
        raise RuntimeError("Legacy red ball is still present in the generated scene")

    articulation_roots = [
        str(prim.GetPath())
        for prim in scene_stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not articulation_roots:
        raise RuntimeError("Referenced Z1 has no ArticulationRootAPI in the task scene")

    return {
        "robot_prim_count": sum(1 for _ in robot_stage.Traverse()),
        "robot_mesh_count_including_instances": mesh_count,
        "robot_collision_count_including_instances": collision_count,
        "scene_prim_count": sum(1 for _ in scene_stage.Traverse()),
        "training_object_mesh_count": training_object_mesh_count,
        "articulation_roots": articulation_roots,
        "color_camera": str(color.GetPath()),
        "depth_camera": str(find_prim_by_name(robot_stage, "DepthCamera").GetPath()),
        "vla_color_camera": str(
            find_prim_by_name(robot_stage, "VLAColorCamera").GetPath()
        ),
        "global_camera": GLOBAL_CAMERA_PATH,
    }


def main() -> None:
    print(f"[BUILD] Importing URDF: {ARGS.urdf}", flush=True)
    try:
        resolved_urdf = resolve_package_mesh_uris(
            ARGS.urdf,
            ARGS.resolved_urdf,
            ARGS.z1_description_root,
        )
        color_path, depth_path, vla_color_path = import_robot(
            resolved_urdf, ARGS.robot_usd
        )
        print(f"[BUILD] Imported robot USD: {ARGS.robot_usd}", flush=True)
        metadata = build_task_scene(
            ARGS.robot_usd,
            ARGS.scene_usd,
            color_path,
            depth_path,
            vla_color_path,
        )
        print(f"[BUILD] Authored task scene: {ARGS.scene_usd}", flush=True)
        validation = validate_outputs(ARGS.robot_usd, ARGS.scene_usd)
        metadata["validation"] = validation
        ARGS.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "ok", **validation}, indent=2))
        print(f"Robot USD: {ARGS.robot_usd}")
        print(f"Scene USD: {ARGS.scene_usd}")
        print(f"Metadata: {ARGS.metadata}")
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        APP.close()


print(f"[BUILD] Module name: {__name__}", flush=True)
main()
