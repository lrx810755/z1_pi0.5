from pathlib import Path
import hashlib,json
from pxr import Usd,UsdGeom,UsdPhysics,Sdf
r=Path(__file__).parent
base=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd')
scene=base.with_name('smolvla_cup_only_frustum_20261025.usd')
assert not scene.exists()
s=Usd.Stage.Open(str(base));s.GetRootLayer().Export(str(scene));s=Usd.Stage.Open(str(scene))
removed=[]
for name in ['Apple','Peach','Mouse','BeverageBottle']:
 path='/World/Task/Objects/'+name;assert s.GetPrimAtPath(path).IsValid();assert s.RemovePrim(path);removed.append(path)
assert [p.GetName() for p in s.GetPrimAtPath('/World/Task/Objects').GetChildren()]==['Cup']
col=s.GetPrimAtPath('/World/Task/Objects/Cup/Collider');assert col.IsA(UsdGeom.Mesh)
assert UsdPhysics.MeshCollisionAPI(col).GetApproximationAttr().Get()=='convexHull'
assert len(UsdGeom.Mesh(col).GetPointsAttr().Get())==128
assert s.GetPrimAtPath('/World/Z1/root_joint/vla_camera_color_optical_frame/VLAColorCamera').IsValid()
s.GetRootLayer().Save()
# Compare authored scene layers after deleting the same subtrees from both.
layers=[]
for path in [base,scene]:
 original=Sdf.Layer.FindOrOpen(str(path));layer=Sdf.Layer.CreateAnonymous('compare.usda');layer.ImportFromString(original.ExportToString())
 for deleted in removed:
  if layer.GetPrimAtPath(deleted):
   edit=Sdf.BatchNamespaceEdit();edit.Add(deleted,Sdf.Path.emptyPath);assert layer.Apply(edit)
 layers.append(layer.ExportToString())
assert layers[0]==layers[1]
out={'verified':True,'scene':str(scene),'baseline_scene':str(base),'sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),'removed_subtrees':removed,'remaining_task_objects':['Cup'],'other_scene_content_identical':True}
(r/'scene_verified.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
