from pathlib import Path
r=Path(__file__).parent;s=(r/'depth_probe.py').read_text().replace("R=G/'depth_probe'","R=G/'material_probe'")
needle=" material=UsdShade.Material.Define(stage,'/World/Looks/CupOpaqueDiagnostic')"
insert=""" # Replace only the glass and condensation bindings with physical MDL glass.
 for mesh in Usd.PrimRange(cp,Usd.TraverseInstanceProxies()):
  if not mesh.IsA(UsdGeom.Mesh):continue
  old=UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
  old_name=old.GetPrim().GetName() if old else ''
  if old_name not in ('Glass','Condensation'):continue
  path='/World/Looks/Physical'+old_name
  mat=UsdShade.Material.Define(stage,path);shader=UsdShade.Shader.Define(stage,path+'/Shader');shader.CreateImplementationSourceAttr('sourceAsset');shader.SetSourceAsset('OmniGlass.mdl','mdl');shader.SetSourceAssetSubIdentifier('OmniGlass','mdl');shader.CreateInput('glass_color',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.98,.99,1));shader.CreateInput('glass_ior',Sdf.ValueTypeNames.Float).Set(1.491);shader.CreateInput('frosting_roughness',Sdf.ValueTypeNames.Float).Set(.025 if old_name=='Glass' else .12);shader.CreateInput('thin_walled',Sdf.ValueTypeNames.Bool).Set(True);shader.CreateOutput('out',Sdf.ValueTypeNames.Token);mat.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out');UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
 capture('mdl_glass')
"""
assert needle in s;s=s.replace(needle,insert+needle)
(r/'material_probe.py').write_text(s)
