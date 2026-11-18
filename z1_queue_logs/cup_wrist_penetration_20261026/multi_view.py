from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','disable_viewport_updates':True})
import omni.usd,omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf,Gf
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.stage import open_stage,is_stage_loading
import numpy as np,json,xml.etree.ElementTree as ET,os
from PIL import Image
from pathlib import Path
G=Path(__file__).parent;R=G/os.environ.get('PROBE_OUT','multi_view');R.mkdir(exist_ok=True)
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
 a=json.loads(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19/metrics.json').read_text());trial=int(os.environ.get('CUP_TRIAL','5'));ep=next(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/batch_01').glob(f'*/episodes/cup_{trial:03d}_*'));e=dict(episode=trial,path=str(ep));tr=[json.loads(l) for l in (Path(e['path'])/'trace.jsonl').read_text().splitlines()];t=min(tr,key=lambda x:abs(x['simulation_elapsed_s']-float(os.environ.get('CUP_TIME','56.93'))));s=t['contact_status']
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
 import sys,carb
 sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts')
 from render_occlusion import configure_close_range_refraction
 products=[]
 configs=json.loads(Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/external_cameras.json').read_text())
 for name,cfg in configs.items():
  cam=UsdGeom.Camera.Define(stage,'/World/Probe_'+name);cam.CreateClippingRangeAttr(Gf.Vec2f(.05,5));cam.CreateHorizontalApertureAttr(20.955);cam.CreateVerticalApertureAttr(20.955*cfg['height']/cfg['width']);cam.CreateFocalLengthAttr(cfg['fx']*20.955/cfg['width']);mat=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),180))*qm(cfg['orientation_xyzw']);mat.SetTranslateOnly(Gf.Vec3d(*cfg['position_m']));UsdGeom.Xformable(cam).AddTransformOp().Set(mat)
 for name,path,resolution in [('wrist','/World/ProbeCamera',(640,360))]+[(name,'/World/Probe_'+name,(cfg['width'],cfg['height'])) for name,cfg in configs.items()]:
  product=rep.create.render_product(path,resolution);rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);products.append((name,rgb))
 def capture(label):
  for _ in range(60):app.update()
  rep.orchestrator.step(rt_subframes=32)
  for name,ann in products:Image.fromarray(ann.get_data()[:,:,:3]).save(R/(label+'_'+name+'.png'))
  print('SAVED_MULTIVIEW',label,flush=True)
 capture('before')
 val=configure_close_range_refraction(carb.settings.get_settings());(R/'renderer_settings.json').write_text(json.dumps(val,indent=2));capture('after')
 (R/'pose.json').write_text(json.dumps(dict(trial=e['episode'],sim_time=t['simulation_elapsed_s']),indent=2))
except Exception:
 import traceback;traceback.print_exc()
finally:app.close()
