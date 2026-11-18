from pathlib import Path
r=Path(__file__).parent;s=(r/'material_probe.py').read_text().replace("R=G/'material_probe'","R=G/'variant_probe'")
needle=" capture('mdl_glass')\n"
insert=""" capture('mdl_glass')
 # Rebind the two opaque contents using MDL too.
 for mesh in Usd.PrimRange(cp,Usd.TraverseInstanceProxies()):
  if not mesh.IsA(UsdGeom.Mesh):continue
  old=UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0];name=old.GetPrim().GetName() if old else ''
  if name not in ('Material','Liquid'):continue
  path='/World/Looks/Native'+name;mat=UsdShade.Material.Define(stage,path);sh=UsdShade.Shader.Define(stage,path+'/Shader');sh.CreateImplementationSourceAttr('sourceAsset');sh.SetSourceAsset('OmniPBR.mdl','mdl');sh.SetSourceAssetSubIdentifier('OmniPBR','mdl');sh.CreateInput('diffuse_color_constant',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.8,.8,.8) if name=='Material' else Gf.Vec3f(.41448173,.1839278,.057545334));sh.CreateInput('reflection_roughness_constant',Sdf.ValueTypeNames.Float).Set(.25 if name=='Material' else .05);sh.CreateInput('enable_opacity',Sdf.ValueTypeNames.Bool).Set(False);sh.CreateInput('opacity_constant',Sdf.ValueTypeNames.Float).Set(1.);sh.CreateOutput('out',Sdf.ValueTypeNames.Token);mat.CreateSurfaceOutput('mdl').ConnectToSource(sh.ConnectableAPI(),'out');UsdShade.MaterialBindingAPI.Apply(mesh).Bind(mat)
 capture('all_mdl')
 glass_meshes=[]
 for mesh in Usd.PrimRange(cp,Usd.TraverseInstanceProxies()):
  if mesh.IsA(UsdGeom.Mesh) and any(x in str(UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath()) for x in ['PhysicalGlass','PhysicalCondensation']):
   glass_meshes.append(mesh);UsdGeom.Imageable(mesh).GetVisibilityAttr().Set('invisible')
 capture('content_only')
 for mesh in glass_meshes:UsdGeom.Imageable(mesh).GetVisibilityAttr().Set('inherited')
 import carb
 settings=carb.settings.get_settings();settings.set('/rtx/translucency/enabled',False);capture('translucency_off');settings.set('/rtx/translucency/enabled',True)
 settings.set('/rtx/rendermode','PathTracing');settings.set('/rtx/pathtracing/spp',32);settings.set('/rtx/pathtracing/totalSpp',32);capture('path_tracing');settings.set('/rtx/rendermode','RaytracedLighting')
"""
assert needle in s;s=s.replace(needle,insert);(r/'variant_probe.py').write_text(s)
