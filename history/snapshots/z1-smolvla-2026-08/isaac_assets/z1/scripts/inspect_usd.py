#!/usr/bin/env python3
"""Inspect a USD asset with Isaac Sim's bundled Python.

Run with Isaac Sim's Python after sourcing ``scripts/setup_env.sh``.
"""

from __future__ import annotations

import argparse
import os
from urllib.parse import unquote, urlparse
from collections import Counter
from pathlib import Path

from project_paths import ASSET_ROOT


DEFAULT_USD = str(
    ASSET_ROOT / "generated_dual_camera" / "smolvla_dual_wrist_camera.usd"
)


def _asset_to_path(asset_path: str, layer_dir: Path) -> Path | None:
    if not asset_path:
        return None
    if asset_path.startswith("file:"):
        parsed = urlparse(asset_path)
        path = Path(unquote(parsed.path))
        return path
    if "://" in asset_path or asset_path.startswith("omniverse:"):
        return None
    path = Path(asset_path)
    if not path.is_absolute():
        path = layer_dir / path
    return path


def _iter_authored_asset_paths(prim):
    for attr in prim.GetAttributes():
        if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
            value = attr.Get()
            if value:
                yield attr.GetName(), value.path
        elif attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray:
            value = attr.Get()
            if value:
                for item in value:
                    yield attr.GetName(), item.path


def inspect_usd(usd_path: str) -> int:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    usd_path = os.path.abspath(os.path.expanduser(usd_path))
    if not os.path.exists(usd_path):
        print(f"ERROR: USD file does not exist: {usd_path}")
        return 2

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"ERROR: failed to open USD stage: {usd_path}")
        return 2

    root_layer = stage.GetRootLayer()
    layer_dir = Path(root_layer.realPath or usd_path).parent
    default_prim = stage.GetDefaultPrim()

    print("========== Stage ==========")
    print(f"file: {usd_path}")
    print(f"root layer: {root_layer.identifier}")
    print(f"default prim: {default_prim.GetPath() if default_prim else '<none>'}")
    print(f"up axis: {UsdGeom.GetStageUpAxis(stage)}")
    print(f"meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")

    prims = list(stage.Traverse())
    type_counts = Counter(prim.GetTypeName() or "<untyped>" for prim in prims)
    print("\n========== Prim Types ==========")
    print(f"total prims: {len(prims)}")
    for type_name, count in type_counts.most_common(20):
        print(f"{type_name}: {count}")

    print("\n========== Physics ==========")
    articulation_roots = []
    rigid_bodies = []
    collisions = []
    joints = []

    for prim in prims:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim.GetPath().pathString)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim.GetPath().pathString)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collisions.append(prim.GetPath().pathString)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)

    print(f"articulation roots: {len(articulation_roots)}")
    for path in articulation_roots:
        print(f"  - {path}")
    print(f"rigid bodies: {len(rigid_bodies)}")
    print(f"collision prims: {len(collisions)}")
    print(f"joints: {len(joints)}")
    for joint in joints:
        type_name = joint.GetTypeName()
        path = joint.GetPath().pathString
        body0 = joint.GetRelationship("physics:body0").GetTargets()
        body1 = joint.GetRelationship("physics:body1").GetTargets()
        print(f"  - {path} [{type_name}] body0={body0} body1={body1}")

    print("\n========== References / Payloads ==========")
    ref_count = 0
    payload_count = 0
    for prim in prims:
        stack = prim.GetPrimStack()
        for spec in stack:
            for ref in spec.referenceList.GetAddedOrExplicitItems():
                ref_count += 1
                print(f"ref  {prim.GetPath()}: asset={ref.assetPath or '<internal>'} prim={ref.primPath}")
            for payload in spec.payloadList.GetAddedOrExplicitItems():
                payload_count += 1
                print(f"payl {prim.GetPath()}: asset={payload.assetPath or '<internal>'} prim={payload.primPath}")
    if ref_count == 0 and payload_count == 0:
        print("no authored references or payloads found")

    print("\n========== Authored Asset Paths ==========")
    missing = []
    found_any = False
    for prim in prims:
        for attr_name, asset_path in _iter_authored_asset_paths(prim):
            found_any = True
            resolved = _asset_to_path(asset_path, layer_dir)
            if resolved is None:
                status = "external"
            else:
                status = "OK" if resolved.exists() else "MISSING"
                if status == "MISSING":
                    missing.append((prim.GetPath().pathString, attr_name, asset_path))
            print(f"{status:8} {prim.GetPath()}.{attr_name} -> {asset_path}")
    if not found_any:
        print("no authored asset-valued attributes found")

    if missing:
        print("\nERROR: missing local asset paths:")
        for prim_path, attr_name, asset_path in missing:
            print(f"  - {prim_path}.{attr_name}: {asset_path}")
        return 1

    print("\nUSD inspection completed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", nargs="?", default=DEFAULT_USD, help="USD/USDZ/USDA/USDC file to inspect")
    parser.add_argument(
        "--no-kit",
        action="store_true",
        help="Do not initialize Isaac Sim/Kit before importing pxr modules",
    )
    args = parser.parse_args()

    simulation_app = None
    if not args.no_kit:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": True})

    try:
        return inspect_usd(args.usd)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
