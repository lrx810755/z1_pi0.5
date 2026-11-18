"""Set only the cup physics collider, before physics initialization."""
import os,json
from pathlib import Path
from experiment_core import geometry

def configure_experiment_cup(stage):
 from pxr import UsdGeom,UsdPhysics,UsdShade
 from cup_collision import define_cup_collider
 shape=os.environ['CUP_EXPERIMENT_SHAPE'];spec=geometry(shape)
 path='/World/Task/Objects/Cup/Collider';prim=stage.GetPrimAtPath(path)
 material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')[0]
 stage.RemovePrim(path)
 if shape=='frustum':prim=define_cup_collider(stage,path,material if material else None)
 else:
  cylinder=UsdGeom.Cylinder.Define(stage,path);cylinder.CreateRadiusAttr(spec['radius_m']);cylinder.CreateHeightAttr(spec['height_m']);cylinder.CreateAxisAttr('Z');cylinder.AddTranslateOp().Set((0,0,spec['height_m']/2))
  prim=cylinder.GetPrim();UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
  UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
  if material:UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
 root=stage.GetPrimAtPath('/World/Task/Objects/Cup')
 payload=dict(expected=spec,prim_type=prim.GetTypeName(),attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes()},mass_kg=UsdPhysics.MassAPI(root).GetMassAttr().Get(),visual='unchanged original cup scan')
 Path(os.environ['CUP_EXPERIMENT_GROUP_DIR'],'isaac_geometry.json').write_text(json.dumps(payload,indent=2))
 assert prim.GetTypeName()==('Mesh' if shape=='frustum' else 'Cylinder')
