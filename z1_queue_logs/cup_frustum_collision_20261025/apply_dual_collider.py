"""Replace the active cup collider without reimporting the robot or scene."""
import sys,json,hashlib,os
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf
root=Path('/home/lp531/ros2_ws/src/z1_ros2')
sys.path.insert(0,str(root/'isaac_assets/z1/scripts'))
from cup_collision import CUP_COLLISION,define_cup_collider
out=Path(__file__).parent
scene=root/'isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd'
collider='/World/Task/Objects/Cup/Collider'
stage=Usd.Stage.Open(str(scene));old=stage.GetPrimAtPath(collider)
assert old.IsA(UsdGeom.Cylinder),'Unexpected collider before migration'
physics_targets=UsdShade.MaterialBindingAPI(old).GetDirectBindingRel('physics').GetTargets()
visual_targets=UsdShade.MaterialBindingAPI(old).GetDirectBindingRel().GetTargets()
assert len(physics_targets)==1
original=stage.GetRootLayer().ExportToString()
comparison_before=Sdf.Layer.CreateAnonymous("comparison.usda");comparison_before.ImportFromString(original)
# Copy the root into a temporary file beside the scene so relative assets resolve unchanged.
temp=scene.with_name('smolvla_dual_wrist_camera.frustum_pending.usd')
stage.GetRootLayer().Export(str(temp))
s=Usd.Stage.Open(str(temp));s.RemovePrim(collider)
p=define_cup_collider(s,collider,UsdShade.Material(s.GetPrimAtPath(physics_targets[0])))
if visual_targets:
 UsdShade.MaterialBindingAPI(p).Bind(UsdShade.Material(s.GetPrimAtPath(visual_targets[0])))
s.GetRootLayer().Save()
reopened=Usd.Stage.Open(str(temp));p=reopened.GetPrimAtPath(collider)
assert reopened.GetPrimAtPath('/World/Z1/root_joint/vla_camera_color_optical_frame/VLAColorCamera').IsValid()
assert p.IsA(UsdGeom.Mesh)
assert UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()=='convexHull'
assert len(UsdGeom.Mesh(p).GetPointsAttr().Get())==128
assert UsdShade.MaterialBindingAPI(p).GetDirectBindingRel('physics').GetTargets()==physics_targets
# All authored scene content outside the replaced collider must remain byte-identical as USDA.
comparison_after=Sdf.Layer.CreateAnonymous("comparison.usda");comparison_after.ImportFromString(reopened.GetRootLayer().ExportToString())
for layer in (comparison_before,comparison_after):
 scratch=Usd.Stage.Open(layer);scratch.RemovePrim(collider)
assert comparison_before.ExportToString()==comparison_after.ExportToString(),'Unrelated scene edits detected'
os.replace(temp,scene)
for rel in ['isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.json']:
 path=root/rel;data=json.loads(path.read_text())
 if 'initial_state' in data:data['initial_state']['objects']['cup']['collision']=dict(CUP_COLLISION)
 else:
  cup=next(a for a in data['assets'] if a['name']=='cup');cup['collision']=dict(CUP_COLLISION)
 path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
report={'applied':True,'scene':str(scene),'collider':collider,'geometry':CUP_COLLISION,'vertices':128,'faces':66,'physics_material':str(physics_targets[0]),'other_scene_content_identical':True,'sha256':hashlib.sha256(scene.read_bytes()).hexdigest()}
(out/'asset_verified.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
