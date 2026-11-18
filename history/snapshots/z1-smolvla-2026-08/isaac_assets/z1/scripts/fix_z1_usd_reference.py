from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import shutil
from pxr import Usd, Sdf

from project_paths import ASSET_ROOT

USD_PATH = str(ASSET_ROOT / "z1.usd")
BACKUP_PATH = str(ASSET_ROOT / "z1.usd.bak")

OLD_TARGET = Sdf.Path("/visuals/world")
NEW_TARGET = Sdf.Path("/visuals")

if not os.path.exists(USD_PATH):
    raise FileNotFoundError(USD_PATH)

if not os.path.exists(BACKUP_PATH):
    shutil.copy2(USD_PATH, BACKUP_PATH)
    print("Backup created:", BACKUP_PATH)
else:
    print("Backup already exists:", BACKUP_PATH)

stage = Usd.Stage.Open(USD_PATH)
if stage is None:
    raise RuntimeError(f"Cannot open USD stage: {USD_PATH}")

print("\n========== Before ==========")
print("DefaultPrim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else "None")

changed = False

for prim in stage.Traverse():
    refs = prim.GetMetadata("references")
    if not refs:
        continue

    ref_list = refs.GetAddedOrExplicitItems()

    if not ref_list:
        continue

    new_refs = []
    prim_changed = False

    for ref in ref_list:
        asset_path = ref.assetPath
        prim_path = ref.primPath

        print("Found reference on:", prim.GetPath())
        print("  assetPath:", asset_path)
        print("  primPath :", prim_path)

        if prim_path == OLD_TARGET:
            print("  FIX:", OLD_TARGET, "->", NEW_TARGET)
            new_ref = Sdf.Reference(
                assetPath=asset_path,
                primPath=NEW_TARGET,
                layerOffset=ref.layerOffset,
                customData=ref.customData
            )
            new_refs.append(new_ref)
            prim_changed = True
            changed = True
        else:
            new_refs.append(ref)

    if prim_changed:
        prim.GetReferences().ClearReferences()
        for new_ref in new_refs:
            prim.GetReferences().AddReference(new_ref)

# 设置 defaultPrim = /World
world_prim = stage.GetPrimAtPath("/World")
if not world_prim or not world_prim.IsValid():
    print("ERROR: /World 不存在，当前根 prim 为：")
    for p in stage.GetPseudoRoot().GetChildren():
        print(" -", p.GetPath(), p.GetTypeName())
    simulation_app.close()
    raise SystemExit

stage.SetDefaultPrim(world_prim)

stage.GetRootLayer().Save()

print("\n========== After ==========")
stage2 = Usd.Stage.Open(USD_PATH)

default_prim = stage2.GetDefaultPrim()
print("DefaultPrim:", default_prim.GetPath() if default_prim else "None")

check_paths = [
    "/World",
    "/World/z1_description",
    "/World/z1_description/world/visuals",
    "/World/z1_description/world/visuals/link00",
]

for path in check_paths:
    prim = stage2.GetPrimAtPath(path)
    print(path, "valid", bool(prim and prim.IsValid()))

print("\nChanged reference:", changed)
print("Saved:", USD_PATH)

simulation_app.close()
