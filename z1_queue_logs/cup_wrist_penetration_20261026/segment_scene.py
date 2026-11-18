from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','disable_viewport_updates':True})
import omni.usd,omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.stage import open_stage,is_stage_loading
import numpy as np,json,xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path
R=Path(__file__).parent
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
 a=json.loads(Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/failure_analysis/never_close19/metrics.json').read_text());e=next(x for x in a if x['episode']==18);tr=[json.loads(l) for l in (Path(e['path'])/'trace.jsonl').read_text().splitlines()];t=min(tr,key=lambda x:abs(x['simulation_elapsed_s']-24));s=t['contact_status']
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
 camera=UsdGeom.Camera.Define(stage,'/World/ProbeCamera');camera.CreateClippingRangeAttr(Gf.Vec2f(.02,5));camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(20.955*360/640);camera.CreateFocalLengthAttr(250*20.955/640);local=Gf.Matrix4d(*np.array(json.loads((R/'geometry.json').read_text())['camera']['matrix_row_vector']).flatten());UsdGeom.Xformable(camera).AddTransformOp().Set(local*measured)
 rp=rep.create.render_product('/World/ProbeCamera',(640,360));rgb=rep.AnnotatorRegistry.get_annotator('rgb');sem=rep.AnnotatorRegistry.get_annotator('semantic_segmentation',init_params={'colorize':False});rgb.attach([rp]);sem.attach([rp])
 for _ in range(45):app.update()
 rep.orchestrator.step(rt_subframes=32);im=rgb.get_data();seg=sem.get_data();Image.fromarray(im[:,:,:3]).save(R/'full_scene_trial18.png');np.save(R/'semantic18.npy',seg['data']);(R/'semantic18.json').write_text(json.dumps(seg['info'],indent=2));print('LABELS',seg['info'],flush=True)
 # Hide cup in the same frozen pose to identify the grey region beneath its image.
 UsdGeom.Imageable(cp).GetVisibilityAttr().Set('invisible')
 for _ in range(45):app.update()
 rep.orchestrator.step(rt_subframes=32);im=rgb.get_data();seg=sem.get_data();Image.fromarray(im[:,:,:3]).save(R/'full_scene_trial18_cup_hidden.png');np.save(R/'semantic18_cup_hidden.npy',seg['data']);(R/'semantic18_cup_hidden.json').write_text(json.dumps(seg['info'],indent=2));print('DONE',flush=True)
 UsdGeom.Imageable(stage.GetPrimAtPath('/World/Z1')).GetVisibilityAttr().Set('invisible')
 for _ in range(45):app.update()
 rep.orchestrator.step(rt_subframes=32);im=rgb.get_data();Image.fromarray(im[:,:,:3]).save(R/'full_scene_trial18_robot_hidden.png');print('ROBOT_HIDDEN',flush=True)
except Exception:
 import traceback;traceback.print_exc()
finally:app.close()
