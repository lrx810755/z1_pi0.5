#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from isaacsim import SimulationApp
from z1_paths import asset_root


simulation_app = None
Gf = None
Usd = None
UsdGeom = None
UsdShade = None


def start_simulation_app():
    global simulation_app, Gf, Usd, UsdGeom, UsdShade
    if simulation_app is not None:
        return
    simulation_app = SimulationApp({"headless": True})
    from pxr import Gf as _Gf, Usd as _Usd, UsdGeom as _UsdGeom, UsdShade as _UsdShade

    Gf = _Gf
    Usd = _Usd
    UsdGeom = _UsdGeom
    UsdShade = _UsdShade


RED_THRESHOLD = 0.65
NON_RED_LIMIT = 0.35


def _is_red_color(value):
    if value is None:
        return False
    if isinstance(value, Gf.Vec3f) or isinstance(value, Gf.Vec3d):
        colors = [value]
    else:
        colors = list(value)
    for color in colors:
        if len(color) >= 3 and float(color[0]) >= RED_THRESHOLD and float(color[1]) <= NON_RED_LIMIT and float(color[2]) <= NON_RED_LIMIT:
            return True
    return False


def _iter_red_prims(stage):
    for prim in stage.Traverse():
        attr = prim.GetAttribute("primvars:displayColor")
        if attr and _is_red_color(attr.Get()):
            yield prim


def _material_color(prim):
    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not bound:
        return None
    for shader in UsdShade.Material(bound).ComputeSurfaceSource()[0:1]:
        if shader:
            for name in ("inputs:diffuse_color", "inputs:diffuseColor", "inputs:color"):
                attr = shader.GetPrim().GetAttribute(name)
                if attr:
                    value = attr.Get()
                    if value is not None:
                        return value
    return None


def _looks_like_camera_visual(path):
    text = path.lower()
    return "camera" in text or "wrist" in text or "visible" in text


def list_camera_prims(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {usd_path}")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not _looks_like_camera_visual(path):
            continue
        color_attr = prim.GetAttribute("primvars:displayColor")
        color = color_attr.Get() if color_attr else None
        material_color = _material_color(prim)
        imageable = UsdGeom.Imageable(prim)
        visibility = imageable.GetVisibilityAttr().Get() if imageable else None
        print(f"{path} type={prim.GetTypeName()} visibility={visibility} displayColor={color} materialColor={material_color}")


def list_red_prims(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {usd_path}")
    for prim in _iter_red_prims(stage):
        imageable = UsdGeom.Imageable(prim)
        visibility = imageable.GetVisibilityAttr().Get() if imageable else None
        print(f"{prim.GetPath()} type={prim.GetTypeName()} visibility={visibility}")


def hide_red_prims(usd_path):
    backup_path = usd_path.with_suffix(usd_path.suffix + ".red_backup")
    if not backup_path.exists():
        shutil.copy2(usd_path, backup_path)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {usd_path}")

    changed = []
    for prim in _iter_red_prims(stage):
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        changed.append(prim.GetPath().pathString)

    if changed:
        stage.GetRootLayer().Save()
    print(f"backup: {backup_path}")
    print(f"hidden_count: {len(changed)}")
    for path in changed:
        print(path)


def rewrite_camera_visuals(usd_path):
    backup_path = usd_path.with_suffix(usd_path.suffix + ".camera_visual_backup")
    if not backup_path.exists():
        shutil.copy2(usd_path, backup_path)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {usd_path}")
    stage.Load()

    changed = []
    for prim in list(stage.Traverse()):
        if prim.GetName() != "WristCamera":
            continue
        camera_path = prim.GetPath()
        for child in list(prim.GetChildren()):
            if child.GetName().startswith("VisibleCamera"):
                stage.RemovePrim(child.GetPath())
                changed.append(f"removed {child.GetPath()}")

        body_path = camera_path.AppendChild("VisibleCameraBody")
        body = UsdGeom.Cube.Define(stage, body_path)
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        body.AddScaleOp().Set(Gf.Vec3f(0.012, 0.010, 0.008))
        body.GetDisplayColorAttr().Set([Gf.Vec3f(0.92, 0.92, 0.88)])
        changed.append(f"created {body_path}")

    if changed:
        stage.GetRootLayer().Save()
    print(f"backup: {backup_path}")
    print(f"changed_count: {len(changed)}")
    for item in changed:
        print(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "usd", type=Path, default=asset_root() / "z1.usd", nargs="?"
    )
    parser.add_argument("--hide", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--rewrite-camera-visuals", action="store_true")
    args = parser.parse_args()

    start_simulation_app()

    if args.rewrite_camera_visuals:
        rewrite_camera_visuals(args.usd)
    elif args.camera:
        list_camera_prims(args.usd)
    elif args.hide:
        hide_red_prims(args.usd)
    else:
        list_red_prims(args.usd)


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
