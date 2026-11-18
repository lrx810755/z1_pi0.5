from pathlib import Path
import json,hashlib
from pxr import Usd,UsdGeom,UsdPhysics,Sdf
R=Path(__file__).parent
source=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd')
backup=Path('/home/lp531/z1_queue_logs/cup_frustum_collision_20261023/backup/isaac_assets/z1/generated/smolvla_red_ball_into_box.usd')
target=source.with_name('smolvla_cup_only_cylinder_compare_20261026.usd');assert not target.exists()
path='/World/Task/Objects/Cup/Collider';original=Sdf.Layer.FindOrOpen(str(backup));old=original.GetPrimAtPath(path);assert old.typeName=='Cylinder'
s=Usd.Stage.Open(str(source));before=s.GetRootLayer().ExportToString();s.GetRootLayer().Export(str(target));t=Usd.Stage.Open(str(target));t.RemovePrim(path);assert Sdf.CopySpec(original,path,t.GetRootLayer(),path);t.GetRootLayer().Save()
p=t.GetPrimAtPath(path);assert p.IsA(UsdGeom.Cylinder);c=UsdGeom.Cylinder(p);assert c.GetRadiusAttr().Get()==.0252 and c.GetHeightAttr().Get()==.061
layers=[]
for text in [before,t.GetRootLayer().ExportToString()]:
 layer=Sdf.Layer.CreateAnonymous("compare.usda");assert layer.ImportFromString(text);stage=Usd.Stage.Open(layer);stage.RemovePrim(path);layers.append(layer.ExportToString())
assert layers[0]==layers[1],'Unexpected changes outside Cup/Collider'
manifest=dict(source_scene=str(source),cylinder_scene=str(target),cylinder_prim_source=str(backup),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cylinder_scene_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),only_cup_collider_changed=True,radius_m=c.GetRadiusAttr().Get(),height_m=c.GetHeightAttr().Get(),translate=list(p.GetAttribute('xformOp:translate').Get()),mass_kg=UsdPhysics.MassAPI(t.GetPrimAtPath('/World/Task/Objects/Cup')).GetMassAttr().Get())
(R/'scene_validation.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
