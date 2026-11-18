from pxr import Usd,UsdGeom,UsdShade,UsdPhysics
from pathlib import Path
import json
s=Usd.Stage.Open('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd')
cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy],useExtentsHint=False)
rows=[]
for p in Usd.PrimRange.Stage(s,Usd.TraverseInstanceProxies()):
 path=str(p.GetPath())
 if not any(q in path for q in ['/Objects/Cup','gripperStator','gripperMover','Looks']):continue
 if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Camera) or p.IsA(UsdShade.Shader) or 'physics:collisionEnabled' in [a.GetName() for a in p.GetAttributes()]:
  d=dict(path=path,type=p.GetTypeName(),active=p.IsActive(),attrs={a.GetName():str(a.Get()) for a in p.GetAttributes() if any(q in a.GetName().lower() for q in ['opacity','transmission','thin','collisionenabled','clipping','doublesided','visibility','approximation','filtered','displayopacity'])})
  if p.IsA(UsdGeom.Mesh):
   mesh=UsdGeom.Mesh(p);d['points']=len(mesh.GetPointsAttr().Get() or []);b=cache.ComputeWorldBound(p).ComputeAlignedRange();d['world_bounds']=[list(b.GetMin()),list(b.GetMax())];d['material']=str(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0].GetPath())
  rows.append(d)
Path('/home/lp531/z1_queue_logs/cup_wrist_penetration_20261026/scene_inspection.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
