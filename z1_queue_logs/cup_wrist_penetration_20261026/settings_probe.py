from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','disable_viewport_updates':True})
import omni.usd,omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf,Gf
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.stage import open_stage,is_stage_loading
import numpy as np,json,xml.etree.ElementTree as ET,os
from PIL import Image
from pathlib import Path
G=Path(__file__).parent;R=G/'settings_probe';R.mkdir(exist_ok=True)
try:
 open_stage('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd')
 while is_stage_loading():app.update()
 stage=omni.usd.get_context().get_stage()
 for p in list(stage.Traverse()):
  if p.IsA(UsdPhysics.Joint):p.SetActive(False)
  if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False)
  if p.HasAPI(UsdPhysics.ArticulationRootAPI):p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
 for p in ['/World/Z1/root_joint/vla_camera_link/visuals','/World/Z1/root_joint/vla_camera_link/collisions']:
  if stage.GetPrimAtPath(p):stage.GetPrimAtPath(p).SetActive(False)
 add_update_semantics(stage.GetPrimAtPath('/World/Z1'),'robot')
 add_update_semantics(stage.GetPrimAtPath('/World/Task/WorkSurface'),'table')
 add_update_semantics(stage.GetPrimAtPath('/World/Task/Objects/Cup'),'cup')
 a=json.loads(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19/metrics.json').read_text());e=next(x for x in a if x['episode']==int(os.environ.get('CUP_TRIAL','5')));tr=[json.loads(l) for l in (Path(e['path'])/'trace.jsonl').read_text().splitlines()];t=min(tr,key=lambda x:abs(x['simulation_elapsed_s']-float(os.environ.get('CUP_TIME','56.93'))));s=t['contact_status']
 def qm(q):
  x,y,z,w=q;return Gf.Matrix4d().SetRotate(Gf.Quatd(w,Gf.Vec3d(x,y,z)))
 urdf=ET.parse('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf').getroot();poses={'world':Gf.Matrix4d(1)};pending=list(urdf.findall('joint'));qmap=dict(zip([f'joint{i}' for i in range(1,7)]+['jointGripper'],t['joint_state']))
 while pending:
  progressed=False
  for j in pending[:]:
   parent=j.find('parent').get('link');child=j.find('child').get('link')
   if parent not in poses:continue
   o=j.find('origin');xyz=[float(x) for x in o.get('xyz','0 0 0').split()] if o is not None else [0]*3;rpy=[float(x) for x in o.get('rpy','0 0 0').split()] if o is not None else [0]*3
   rot=Gf.Rotation(Gf.Vec3d(0,0,1),np.degrees(rpy[2]))*Gf.Rotation(Gf.Vec3d(0,1,0),np.degrees(rpy[1]))*Gf.Rotation(Gf.Vec3d(1,0,0),np.degrees(rpy[0]));m=Gf.Matrix4d().SetRotate(rot);m.SetTranslateOnly(Gf.Vec3d(*xyz));axis=j.find('axis');angle=qmap.get(j.get('name'),0);r=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*([float(x) for x in axis.get('xyz').split()] if axis is not None else [1,0,0])),np.degrees(angle)));poses[child]=r*m*poses[parent];pending.remove(j);progressed=True
  if not progressed:raise RuntimeError('URDF tree disconnected')
 measured=qm(s['gripper_world_orientation_xyzw']);measured.SetTranslateOnly(Gf.Vec3d(*s['gripper_world_position_m']));adjust=poses['gripperStator'].GetInverse()*measured
 for name,matrix in poses.items():
  p=stage.GetPrimAtPath('/World/Z1/root_joint/'+name)
  if p:
   xf=UsdGeom.Xformable(p);xf.ClearXformOpOrder();xf.AddTransformOp().Set(matrix*adjust)
 cp=stage.GetPrimAtPath('/World/Task/Objects/Cup');matrix=qm(s['object_orientation_xyzw']);matrix.SetTranslateOnly(Gf.Vec3d(*s['object_position_m']));xf=UsdGeom.Xformable(cp);xf.ClearXformOpOrder();xf.AddTransformOp().Set(matrix)
 camera=UsdGeom.Camera.Define(stage,'/World/ProbeCamera');camera.CreateClippingRangeAttr(Gf.Vec2f(.02,5));camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(20.955*360/640);camera.CreateFocalLengthAttr(250*20.955/640);local=Gf.Matrix4d(*np.array(json.loads((G/'geometry.json').read_text())['camera']['matrix_row_vector']).flatten());UsdGeom.Xformable(camera).AddTransformOp().Set(local*measured)
 rp=rep.create.render_product('/World/ProbeCamera',(640,360));rgb=rep.AnnotatorRegistry.get_annotator('rgb');sem=rep.AnnotatorRegistry.get_annotator('semantic_segmentation',init_params={'colorize':False});depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');rgb.attach([rp]);sem.attach([rp]);depth.attach([rp])
 def capture(name):
  for _ in range(60):app.update()
  rep.orchestrator.step(rt_subframes=32)
  im=rgb.get_data();seg=sem.get_data();dep=depth.get_data();assert im.size and seg['data'].size and dep.size
  Image.fromarray(im[:,:,:3]).save(R/(name+'.png'));np.save(R/(name+'_semantic.npy'),seg['data']);np.save(R/(name+'_depth.npy'),dep);(R/(name+'_labels.json')).write_text(json.dumps(seg['info'],indent=2));print('SAVED',name,flush=True)
 capture('original')
 import carb
 settings=carb.settings.get_settings();keys=['/rtx/translucency/worldEps','/rtx/raytracing/fractionalCutoutOpacity','/rtx/material/translucencyAsOpacity','/rtx/translucency/virtualDepth','/rtx/rendermode'];values={k:settings.get(k) for k in keys};print('ORIGINAL_SETTINGS',values,flush=True);(R/'original_settings.json').write_text(json.dumps(values,indent=2))
 for name,key,value in [('world_eps_0p0001','/rtx/translucency/worldEps',.0001),('world_eps_zero','/rtx/translucency/worldEps',0.),('fractional_cutout','/rtx/raytracing/fractionalCutoutOpacity',True),('translucency_as_opacity','/rtx/material/translucencyAsOpacity',True),('virtual_depth','/rtx/translucency/virtualDepth',True)]:
  old=settings.get(key);settings.set(key,value);capture(name)
  if old is not None:settings.set(key,old)
  else:settings.destroy_item(key)
 settings.set('/rtx/rendermode','PathTracing');settings.set('/rtx/pathtracing/spp',32);settings.set('/rtx/pathtracing/totalSpp',32);capture('original_path_tracing');settings.set('/rtx/rendermode','RaytracedLighting')
 robot=stage.GetPrimAtPath('/World/Z1');table=stage.GetPrimAtPath('/World/Task/WorkSurface')
 UsdGeom.Imageable(cp).GetVisibilityAttr().Set('invisible');UsdGeom.Imageable(table).GetVisibilityAttr().Set('invisible');capture('robot_only')
 UsdGeom.Imageable(robot).GetVisibilityAttr().Set('invisible');UsdGeom.Imageable(cp).GetVisibilityAttr().Set('inherited');capture('cup_only')
 UsdGeom.Imageable(robot).GetVisibilityAttr().Set('inherited');UsdGeom.Imageable(table).GetVisibilityAttr().Set('inherited')
 material=UsdShade.Material.Define(stage,'/World/Looks/CupOpaqueDiagnostic');sh=UsdShade.Shader.Define(stage,'/World/Looks/CupOpaqueDiagnostic/Shader');sh.CreateIdAttr('UsdPreviewSurface');sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1,.25,0));sh.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(1);sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.5);material.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(cp).Bind(material,UsdShade.Tokens.strongerThanDescendants)
 capture('opaque_cup')
 report=dict(trial=e['episode'],sim_time=t['simulation_elapsed_s'],trace=t)
 (R/'pose.json').write_text(json.dumps(report,indent=2));print('DONE_DEPTH_PROBE',flush=True)
except Exception:
 import traceback;traceback.print_exc()
finally:app.close()
