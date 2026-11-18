#!/usr/bin/env python3
"""Normalize downloaded high-quality object assets into local USD wrappers.

Run with Isaac Sim's Python:

    /home/lp531/isaac-sim/python.sh \
      /home/lp531/isaac_assets/z1/scripts/prepare_scanned_assets.py

The wrappers keep the original textured geometry as references, convert every
asset to a consistent tabletop scale, center it in XY, and place its lowest
render bound at Z=0.  Physics is authored by the task-scene builder so the
render assets remain reusable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


ROOT = Path("/home/lp531/isaac_assets/z1")
SOURCE_DIR = ROOT / "scanned_objects/source"
OUTPUT_DIR = ROOT / "scanned_objects/usd"
MANIFEST_PATH = ROOT / "scanned_objects/manifest.json"
GRIPPER_RECOMMENDED_MAX_WIDTH_M = 0.055

ASSETS = {
    "cup": {
        "source_file": "cup.usdz",
        "output_file": "cup.usd",
        "scale": 0.50,
        "source_bounds_m": {
            "min": [-0.050291, -0.050291, 0.000060],
            "max": [0.050291, 0.050291, 0.122000],
        },
        "title": "Indian Tea Glass (Chai)",
        "author": "bhagathgoud",
        "license": "CC Attribution",
        "source_url": "https://sketchfab.com/3d-models/indian-tea-glass-chai-bc87e1579d544bbd9f7d132095e40e3d",
        "collision": {"shape": "cylinder", "radius_m": 0.0252, "height_m": 0.061},
        "grasp": {
            "strategy": "side_pinch",
            "width_m": 0.0504,
            "height_m": 0.035,
        },
    },
    "apple": {
        "source_file": "apple.usdz",
        "output_file": "apple.usd",
        "scale": 0.29,
        # Measured directly from the composed render-mesh points.  The former
        # values came from stale USD extent hints and overstated every axis,
        # producing a 52 mm collision proxy around a 36 mm rendered apple.
        "source_bounds_m": {
            "min": [-0.064730033, -0.060439616, 0.002665441],
            "max": [0.060461406, 0.064105600, 0.159896448],
        },
        "title": "Apple",
        "author": "Moshe Caine",
        "license": "CC Attribution",
        "source_url": "https://sketchfab.com/3d-models/apple-da0cfd4c6a4e41c5aaf06797b463235b",
        "collision": {
            "shape": "ellipsoid",
            # Fit the fruit body only.  The rendered scan reaches 45.60 mm
            # because its narrow stem extends from roughly z=38 to 46 mm;
            # including that stem in one convex ellipsoid makes the upper
            # collision silhouette much wider than the visible apple.
            "radii_m": [0.0176, 0.0175, 0.0190],
        },
        "grasp": {
            "strategy": "equatorial_pinch",
            "width_m": 0.0352,
            "height_m": 0.0190,
        },
    },
    "peach": {
        "source_file": "peach.usdz",
        "output_file": "peach.usd",
        "scale": 0.24,
        # Measured from the composed source mesh rather than the stale extent
        # hints.  The old bounds made the normalized render float 8.3 mm above
        # its root and produced a roughly 52 mm proxy around a 31 mm peach.
        "source_bounds_m": {
            "min": [-0.065709144, -0.061423801, -0.001713958],
            "max": [0.063139260, 0.064943366, 0.119007647],
        },
        "title": "Peach Polycam",
        "author": "Moshe Caine",
        "license": "CC Attribution",
        "source_url": "https://sketchfab.com/3d-models/peach-polycam-b6a1d0ee969e4cbd921249be04b16e65",
        "collision": {
            "shape": "ellipsoid",
            "radii_m": [0.01545, 0.01515, 0.01450],
        },
        "grasp": {
            "strategy": "equatorial_pinch",
            "width_m": 0.0309,
            "height_m": 0.0145,
        },
    },
    "mouse": {
        "source_file": "mouse.usdz",
        "output_file": "mouse.usd",
        "scale": 0.44,
        "source_bounds_m": {
            "min": [-0.100000, -0.056114, 0.007768],
            "max": [0.100000, 0.056114, 0.072232],
        },
        "title": "Computer Mouse",
        "author": "kubassa",
        "license": "CC Attribution",
        "source_url": "https://sketchfab.com/3d-models/computer-mouse-4a6e2b1cf5d54ed5b97da093ed1e74a4",
        "collision": {
            "shape": "ellipsoid",
            "radii_m": [0.043, 0.0245, 0.01415],
        },
        "grasp": {
            "strategy": "side_pinch_across_short_axis",
            "width_m": 0.0490,
            "height_m": 0.020,
        },
    },
    "beverage_bottle": {
        "source_file": "beverage_bottle.usdz",
        "output_file": "beverage_bottle.usd",
        "scale": 0.50,
        "source_bounds_m": {
            "min": [-0.048085, -0.048206, -0.000557],
            "max": [0.048085, 0.048477, 0.313394],
        },
        "title": "Coca-Cola Bottle",
        "author": "MemoRorschach",
        "license": "CC Attribution",
        "source_url": "https://sketchfab.com/3d-models/coca-cola-bottle-34075fedb0ef40d9a172231134849914",
        "collision": {"shape": "cylinder", "radius_m": 0.0242, "height_m": 0.157},
        "grasp": {
            "strategy": "side_pinch_on_body",
            "width_m": 0.0484,
            "height_m": 0.075,
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_normalized_bounds(config: dict) -> tuple[list[float], list[float]]:
    minimum = config["source_bounds_m"]["min"]
    maximum = config["source_bounds_m"]["max"]
    scale = config["scale"]
    size = [(maximum[index] - minimum[index]) * scale for index in range(3)]
    return ([-size[0] / 2.0, -size[1] / 2.0, 0.0], [size[0] / 2.0, size[1] / 2.0, size[2]])


def render_point_bounds(
    stage: Usd.Stage, root: Usd.Prim
) -> tuple[list[float], list[float]]:
    """Measure composed render vertices instead of trusting cached extents."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_inverse = xform_cache.GetLocalToWorldTransform(root).GetInverse()
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    point_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        transform = xform_cache.GetLocalToWorldTransform(prim)
        for point in UsdGeom.Mesh(prim).GetPointsAttr().Get() or []:
            local = root_inverse.Transform(transform.Transform(point))
            for axis in range(3):
                minimum[axis] = min(minimum[axis], float(local[axis]))
                maximum[axis] = max(maximum[axis], float(local[axis]))
            point_count += 1
    if point_count == 0:
        raise RuntimeError(f"{root.GetPath()} contains no render mesh points")
    return minimum, maximum


def build_wrapper(name: str, config: dict) -> dict:
    source_path = SOURCE_DIR / config["source_file"]
    output_path = OUTPUT_DIR / config["output_file"]
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    grasp_width = float(config["grasp"]["width_m"])
    if grasp_width > GRIPPER_RECOMMENDED_MAX_WIDTH_M:
        raise RuntimeError(
            f"{name} grasp width {grasp_width:.4f} m exceeds the conservative Z1 "
            f"limit {GRIPPER_RECOMMENDED_MAX_WIDTH_M:.4f} m"
        )
    if output_path.exists():
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())

    normalized = UsdGeom.Xform.Define(stage, "/Asset/Normalized")
    visual = UsdGeom.Xform.Define(stage, "/Asset/Normalized/Visual")
    visual.GetPrim().GetReferences().AddReference(f"../source/{config['source_file']}")
    minimum = config["source_bounds_m"]["min"]
    maximum = config["source_bounds_m"]["max"]
    scale = float(config["scale"])
    center_x = (minimum[0] + maximum[0]) / 2.0
    center_y = (minimum[1] + maximum[1]) / 2.0
    xformable = UsdGeom.Xformable(normalized)
    xformable.AddTranslateOp().Set(
        Gf.Vec3d(-center_x * scale, -center_y * scale, -minimum[2] * scale)
    )
    xformable.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    removed_physics_api_count = 0
    physics_apis = (
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.CollisionAPI,
        UsdPhysics.MassAPI,
        PhysxSchema.PhysxRigidBodyAPI,
        PhysxSchema.PhysxCollisionAPI,
    )
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(visual.GetPath()):
            continue
        for api_schema in physics_apis:
            if prim.HasAPI(api_schema):
                prim.RemoveAPI(api_schema)
                removed_physics_api_count += 1

    root_prim = root.GetPrim()
    root_prim.CreateAttribute("asset:semanticLabel", Sdf.ValueTypeNames.String, custom=True).Set(name)
    root_prim.CreateAttribute("asset:sourceTitle", Sdf.ValueTypeNames.String, custom=True).Set(
        config["title"]
    )
    root_prim.CreateAttribute("asset:sourceAuthor", Sdf.ValueTypeNames.String, custom=True).Set(
        config["author"]
    )
    root_prim.CreateAttribute("asset:license", Sdf.ValueTypeNames.String, custom=True).Set(
        config["license"]
    )
    root_prim.CreateAttribute("asset:sourceUrl", Sdf.ValueTypeNames.String, custom=True).Set(
        config["source_url"]
    )
    root_prim.CreateAttribute(
        "asset:normalizationScale", Sdf.ValueTypeNames.Double, custom=True
    ).Set(scale)

    source_hash = sha256(source_path)
    expected_min, expected_max = expected_normalized_bounds(config)
    stage.GetRootLayer().customLayerData = {
        "assetName": name,
        "sourceFile": f"../source/{config['source_file']}",
        "sourceSha256": source_hash,
        "sourceTitle": config["title"],
        "sourceAuthor": config["author"],
        "sourceLicense": config["license"],
        "sourceUrl": config["source_url"],
        "normalizationScale": scale,
        "removedSourcePhysicsApiCount": removed_physics_api_count,
    }
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(output_path))
    actual_min, actual_max = render_point_bounds(
        reopened, reopened.GetDefaultPrim()
    )
    print(
        f"[ASSET] {name}: bounds min={actual_min} max={actual_max}",
        flush=True,
    )
    for actual, expected in zip(actual_min + actual_max, expected_min + expected_max):
        if abs(actual - expected) > 0.002:
            raise RuntimeError(
                f"{name} normalization mismatch: actual={actual_min, actual_max}, "
                f"expected={expected_min, expected_max}"
            )

    return {
        "name": name,
        "source_file": str(source_path.relative_to(ROOT)),
        "usd_file": str(output_path.relative_to(ROOT)),
        "sha256": source_hash,
        "title": config["title"],
        "author": config["author"],
        "license": config["license"],
        "source_url": config["source_url"],
        "normalization_scale": scale,
        "removed_source_physics_api_count": removed_physics_api_count,
        "bounds_m": {"min": actual_min, "max": actual_max},
        "collision": config["collision"],
        "grasp": {
            **config["grasp"],
            "compatible_with_z1_gripper": True,
            "recommended_max_width_m": GRIPPER_RECOMMENDED_MAX_WIDTH_M,
        },
    }


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--asset", choices=tuple(ASSETS), default="")
        args, _unknown = parser.parse_known_args()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        selected_names = [args.asset] if args.asset else list(ASSETS)
        rebuilt_assets = []
        for name in selected_names:
            config = ASSETS[name]
            print(f"[ASSET] Building {name}", flush=True)
            rebuilt_assets.append(build_wrapper(name, config))
        if args.asset and MANIFEST_PATH.exists():
            previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            rebuilt_by_name = {
                asset["name"]: asset for asset in rebuilt_assets
            }
            assets = [
                rebuilt_by_name.get(asset["name"], asset)
                for asset in previous.get("assets", [])
            ]
            if args.asset not in {asset["name"] for asset in assets}:
                assets.extend(rebuilt_assets)
        else:
            assets = rebuilt_assets
        manifest = {
            "description": "Local high-quality visual assets for the Z1 SmolVLA tabletop scene",
            "license_note": (
                "Upstream credits list each source as CC Attribution; the dataset is "
                "marked cc-by-4.0. Preserve creator attribution when redistributing."
            ),
            "gripper_profile": {
                "model": "Unitree Z1 single-DOF angular gripper",
                "joint_range_deg": [-90.0, 0.0],
                "collision_envelope_m": [0.156, 0.082, 0.06885],
                "recommended_max_object_grasp_width_m": GRIPPER_RECOMMENDED_MAX_WIDTH_M,
                "design_rule": (
                    "Use a regular pinch region no wider than 55 mm; avoid handles "
                    "and strongly concave contact surfaces."
                ),
            },
            "assets": assets,
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        APP.close()


main()
