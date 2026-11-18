from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':640,'height':360,'renderer':'RayTracedLighting','disable_viewport_updates':True})
import omni.usd,omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdLux,UsdShade,UsdPhysics,Gf,Sdf
import numpy as np,json
from pathlib import Path
from PIL import Image
R=Path(__file__).parent
try:
 stage=omni.usd.get_context().get_stage();source='/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd'
 gr=stage.DefinePrim('/World/Gripper','Xform');gr.GetReferences().AddReference(source,'/World/Z1/root_joint/gripperStator/visuals');xf=UsdGeom.Xformable(gr);xf.ClearXformOpOrder();xf.AddTransformOp().Set(Gf.Matrix4d(1))
 cup=stage.DefinePrim('/World/Cup','Xform');cup.GetReferences().AddReference(source,'/World/Task/Objects/Cup')
 a=json.loads(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19/metrics.json').read_text());e=next(x for x in a if x['episode']==8);tr=[json.loads(l) for l in (Path(e['path'])/'trace.jsonl').read_text().splitlines()];s=tr[e['closest_capture']['trace_index']]['contact_status']
 def qmatrix(q):
  x,y,z,w=q;return Gf.Matrix4d().SetRotate(Gf.Quatd(w,Gf.Vec3d(x,y,z)))
 gm=qmatrix(s['gripper_world_orientation_xyzw']);gm.SetTranslateOnly(Gf.Vec3d(*s['gripper_world_position_m']));cm=qmatrix(s['object_orientation_xyzw']);cm.SetTranslateOnly(Gf.Vec3d(*s['object_position_m']));xf=UsdGeom.Xformable(cup);xf.ClearXformOpOrder();xf.AddTransformOp().Set(cm*gm.GetInverse())
 UsdPhysics.RigidBodyAPI(cup).GetRigidBodyEnabledAttr().Set(False)
 material=UsdShade.Material.Define(stage,'/World/Looks/Grey');sh=UsdShade.Shader.Define(stage,'/World/Looks/Grey/Shader');sh.CreateIdAttr('UsdPreviewSurface');sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.3));sh.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(1);material.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
 # Expand imported instances for experimental material and sidedness overrides.
 stack=[gr]
 while stack:
  prim=stack.pop()
  if prim.IsInstance():prim.SetInstanceable(False)
  stack.extend(prim.GetChildren())
 # Referenced robot material is outside subtree; supply the same opaque grey.
 for p in Usd.PrimRange(gr,Usd.TraverseInstanceProxies()):
  if p.IsA(UsdGeom.Mesh) and '/collisions/' not in str(p.GetPath()):UsdShade.MaterialBindingAPI.Apply(p).Bind(material)
 light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(1200)
 camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateClippingRangeAttr(Gf.Vec2f(.02,5));camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(20.955*360/640);camera.CreateFocalLengthAttr(250*20.955/640);xf=UsdGeom.Xformable(camera);xf.AddTransformOp().Set(Gf.Matrix4d(*np.array(json.loads((R/'geometry.json').read_text())['camera']['matrix_row_vector']).flatten()))
 rp=rep.create.render_product('/World/Camera',(640,360));an=rep.AnnotatorRegistry.get_annotator('rgb');an.attach([rp])
 def capture(name):
  for _ in range(45):app.update()
  rep.orchestrator.step(rt_subframes=16);data=an.get_data();assert len(data)>0;Image.fromarray(data[:,:,:3]).save(R/name);print('SAVED',name,'cup_xform',UsdGeom.Xformable(cup).GetLocalTransformation(),flush=True)
 capture('render_baseline.png')
 for p in Usd.PrimRange(gr,Usd.TraverseInstanceProxies()):
  if p.IsA(UsdGeom.Mesh) and '/collisions/' not in str(p.GetPath()):UsdGeom.Mesh(p).GetDoubleSidedAttr().Set(True)
 capture('render_double_sided.png')
 camera.GetClippingRangeAttr().Set(Gf.Vec2f(.001,5));capture('render_double_sided_near1mm.png')
 # Force the cup opaque to test transparent-surface compositing.
 for p in Usd.PrimRange(cup,Usd.TraverseInstanceProxies()):
  if p.IsA(UsdShade.Shader) and p.HasAttribute('inputs:opacity'):p.GetAttribute('inputs:opacity').Set(1.)
 capture('render_cup_opaque.png')
except Exception:
 import traceback;traceback.print_exc();raise
finally:app.close()
