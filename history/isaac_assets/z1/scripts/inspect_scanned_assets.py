#!/usr/bin/env python3
"""Inspect downloaded USD/USDZ training-object assets with Isaac Sim's USD runtime."""

from __future__ import annotations

import json
from pathlib import Path

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdShade


SOURCE_DIR = Path("/home/lp531/isaac_assets/z1/scanned_objects/source")


def inspect_asset(path: Path) -> dict:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Cannot open {path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"{path} has no default prim")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bounds = bbox_cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    size = maximum - minimum
    prims = list(stage.Traverse())
    meshes = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
    materials = [prim for prim in prims if prim.IsA(UsdShade.Material)]
    return {
        "file": path.name,
        "default_prim": str(default_prim.GetPath()),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "bounds": {
            "min": [float(value) for value in minimum],
            "max": [float(value) for value in maximum],
            "size": [float(value) for value in size],
        },
        "prim_count": len(prims),
        "mesh_count": len(meshes),
        "material_count": len(materials),
    }


def main() -> None:
    try:
        results = [inspect_asset(path) for path in sorted(SOURCE_DIR.glob("*.usdz"))]
        print(json.dumps(results, indent=2), flush=True)
    finally:
        APP.close()


main()
