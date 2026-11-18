"""Convert the two default scene assets; preserve all unrelated stage properties."""
import sys,json,hashlib
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
B=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1');sys.path.insert(0,str(B/'scripts'))
from cup_collision import CUP_COLLISION,define_cup_collider,configure_cup_collision
L=Path(__file__).resolve().parent

def signature(stage):
 values=[]
 for p in stage.Traverse():
  if str(p.GetPath()).startswith('/World/Task/Objects/Cup/Collider'):continue
  values.append((str(p.GetPath()),p.GetTypeName(),[(a.GetName(),str(a.Get())) for a in p.GetAttributes()]))
 return hashlib.sha256(repr(values).encode()).hexdigest()
results=[]
for relative in ['generated/smolvla_red_ball_into_box.usd','generated_dual_camera/smolvla_dual_wrist_camera.usd']:
 path=B/relative;stage=Usd.Stage.Open(str(path));before=signature(stage)
 root=stage.GetPrimAtPath('/World/Task/Objects/Cup');mass_before=UsdPhysics.MassAPI(root).GetMassAttr().Get()
 prim=configure_cup_collision(stage);assert prim.IsA(UsdGeom.Cylinder)
 cylinder=UsdGeom.Cylinder(prim);assert cylinder.GetRadiusAttr().Get()==.0252;assert cylinder.GetHeightAttr().Get()==.061
 z=UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()[2];assert abs(z-.0305)<1e-9
 assert signature(stage)==before,'Unrelated stage attributes changed'
 assert UsdPhysics.MassAPI(root).GetMassAttr().Get()==mass_before
 stage.GetRootLayer().Save()
 # Verify saved bytes through a fresh stage and repeat authoring to catch doubled offsets.
 check=Usd.Stage.Open(str(path));prim=configure_cup_collision(check)
 assert abs(UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()[2]-.0305)<1e-9
 assert signature(check)==before
 material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')[0]
 results.append(dict(path=str(path),shape=prim.GetTypeName(),radius_m=.0252,height_m=.061,centre_offset_m=z,mass_kg=mass_before,physics_material=str(material.GetPath()) if material else None,unrelated_attributes_sha256=before))
 meta=path.with_suffix('.json');data=json.loads(meta.read_text());data['initial_state']['objects']['cup']['collision']=dict(CUP_COLLISION);meta.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
manifest=B/'scanned_objects/manifest.json';data=json.loads(manifest.read_text());assets=data['assets']
if isinstance(assets,dict):assets['cup']['collision']=dict(CUP_COLLISION)
else:
 matches=[d for d in assets if d.get('name')=='cup' or d.get('semantic_label')=='cup'];assert len(matches)==1;matches[0]['collision']=dict(CUP_COLLISION)
manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(L/'asset_verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False,indent=2))
