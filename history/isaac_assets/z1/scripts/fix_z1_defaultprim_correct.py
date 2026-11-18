from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf
import os
import shutil

USD_PATH = "/home/lp531/isaac_assets/z1/z1.usd"
BACKUP_PATH = "/home/lp531/isaac_assets/z1/z1.defaultprim_fix_backup.usd"

if not os.path.exists(USD_PATH):
    raise FileNotFoundError(USD_PATH)

if not os.path.exists(BACKUP_PATH):
    shutil.copy2(USD_PATH, BACKUP_PATH)
    print("Backup created:", BACKUP_PATH)

stage = Usd.Stage.Open(USD_PATH)
if stage is None:
    raise RuntimeError("Cannot open USD")

# 1. 修复 </visuals/world> -> </visuals>
old_target = Sdf.Path("/visuals/world")
new_target = Sdf.Path("/visuals")

changed_ref = False

for prim in stage.Traverse():
    refs = prim.GetMetadata("references")
    if not refs:
        continue

    items = refs.GetAddedOrExplicitItems()
    if not items:
        continue

    new_items = []
    prim_changed = False

    for ref in items:
        if ref.primPath == old_target:
            print("Fix reference on:", prim.GetPath())
            print(" ", old_target, "->", new_target)

            new_ref = Sdf.Reference(
                assetPath=ref.assetPath,
                primPath=new_target,
                layerOffset=ref.layerOffset,
                customData=ref.customData,
            )
            new_items.append(new_ref)
            prim_changed = True
            changed_ref = True
        else:
            new_items.append(ref)

    if prim_changed:
        prim.GetReferences().ClearReferences()
        for r in new_items:
            prim.GetReferences().AddReference(r)

# 2. 设置正确的 defaultPrim
robot_prim = stage.GetPrimAtPath("/z1_description")

if not robot_prim or not robot_prim.IsValid():
    print("ERROR: /z1_description 不存在。当前根节点：")
    for p in stage.GetPseudoRoot().GetChildren():
        print(" -", p.GetPath(), p.GetTypeName())
    simulation_app.close()
    raise SystemExit

stage.SetDefaultPrim(robot_prim)
stage.GetRootLayer().Save()

print("\n========== Verify source USD ==========")

stage2 = Usd.Stage.Open(USD_PATH)
default_prim = stage2.GetDefaultPrim()

print("DefaultPrim:", default_prim.GetPath() if default_prim else None)

for path in [
    "/z1_description",
    "/z1_description/world",
    "/z1_description/world/visuals",
    "/z1_description/world/visuals/link00",
    "/visuals",
    "/visuals/link00",
    "/colliders",
    "/meshes",
]:
    prim = stage2.GetPrimAtPath(path)
    print(path, "valid", bool(prim and prim.IsValid()))

print("Changed reference:", changed_ref)
print("Saved:", USD_PATH)

simulation_app.close()

