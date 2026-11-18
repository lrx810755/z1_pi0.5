"""Replay recorded poses with default/fixed RTX settings; no policy or physics replay."""
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','disable_viewport_updates':True})
import sys,os,json,math,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,cv2,carb
from PIL import Image,ImageDraw,ImageFont
import omni.usd,omni.replicator.core as rep
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from isaacsim.core.utils.stage import open_stage,is_stage_loading
R=Path(__file__).parent/'dynamic_replay';R.mkdir(exist_ok=True)
sys.path.insert(0,'/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts')
from render_occlusion import configure_close_range_refraction
try:
 open_stage('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd')
 while is_stage_loading():app.update()
 stage=omni.usd.get_context().get_stage()
 for p in list(stage.Traverse()):
  if p.IsA(UsdPhysics.Joint):p.SetActive(False)
  if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False)
  if p.HasAPI(UsdPhysics.ArticulationRootAPI):p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
 for path in ['/World/Z1/root_joint/vla_camera_link/visuals','/World/Z1/root_joint/vla_camera_link/collisions']:
  p=stage.GetPrimAtPath(path)
  if p:p.SetActive(False)
 def qm(q):
  x,y,z,w=q;return Gf.Matrix4d().SetRotate(Gf.Quatd(w,Gf.Vec3d(x,y,z)))
 urdf=ET.parse('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/z1_with_dual_d435_resolved.urdf').getroot();joints=[]
 for j in urdf.findall('joint'):
  o=j.find('origin');xyz=[float(x) for x in o.get('xyz','0 0 0').split()] if o is not None else [0]*3;rpy=[float(x) for x in o.get('rpy','0 0 0').split()] if o is not None else [0]*3;rot=Gf.Rotation(Gf.Vec3d(0,0,1),np.degrees(rpy[2]))*Gf.Rotation(Gf.Vec3d(0,1,0),np.degrees(rpy[1]))*Gf.Rotation(Gf.Vec3d(1,0,0),np.degrees(rpy[0]));m=Gf.Matrix4d().SetRotate(rot);m.SetTranslateOnly(Gf.Vec3d(*xyz));a=j.find('axis');axis=Gf.Vec3d(*([float(x) for x in a.get('xyz').split()] if a is not None else [1,0,0]));joints.append((j.get('name'),j.find('parent').get('link'),j.find('child').get('link'),m,axis))
 ops={}
 for _,_,child,_,_ in joints:
  p=stage.GetPrimAtPath('/World/Z1/root_joint/'+child)
  if p:
   xf=UsdGeom.Xformable(p);xf.ClearXformOpOrder();ops[child]=xf.AddTransformOp()
 cp=stage.GetPrimAtPath('/World/Task/Objects/Cup');xf=UsdGeom.Xformable(cp);xf.ClearXformOpOrder();cupop=xf.AddTransformOp()
 camera=UsdGeom.Camera.Define(stage,'/World/ReplayWrist');camera.CreateClippingRangeAttr(Gf.Vec2f(.02,5));camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(20.955*360/640);camera.CreateFocalLengthAttr(250*20.955/640);camop=UsdGeom.Xformable(camera).AddTransformOp();local=Gf.Matrix4d(*np.array(json.loads((R.parent/'geometry.json').read_text())['camera']['matrix_row_vector']).flatten())
 def pose(t):
  s=t['contact_status'];qmap=dict(zip([f'joint{i}' for i in range(1,7)]+['jointGripper'],t['joint_state']));poses={'world':Gf.Matrix4d(1)}
  for name,parent,child,m,axis in joints:poses[child]=Gf.Matrix4d().SetRotate(Gf.Rotation(axis,np.degrees(qmap.get(name,0))))*m*poses[parent]
  measured=qm(s['gripper_world_orientation_xyzw']);measured.SetTranslateOnly(Gf.Vec3d(*s['gripper_world_position_m']));adjust=poses['gripperStator'].GetInverse()*measured
  for name,op in ops.items():op.Set(poses[name]*adjust)
  matrix=qm(s['object_orientation_xyzw']);matrix.SetTranslateOnly(Gf.Vec3d(*s['object_position_m']));cupop.Set(matrix);camop.Set(local*measured)
 products=[];cfgs=json.loads(Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/config/external_cameras.json').read_text())
 for name,cfg in cfgs.items():
  cam=UsdGeom.Camera.Define(stage,'/World/Replay_'+name);cam.CreateClippingRangeAttr(Gf.Vec2f(.05,5));cam.CreateHorizontalApertureAttr(20.955);cam.CreateVerticalApertureAttr(20.955*cfg['height']/cfg['width']);cam.CreateFocalLengthAttr(cfg['fx']*20.955/cfg['width']);m=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),180))*qm(cfg['orientation_xyzw']);m.SetTranslateOnly(Gf.Vec3d(*cfg['position_m']));UsdGeom.Xformable(cam).AddTransformOp().Set(m)
 for name,path,size in [('global','/World/Replay_global',(640,480)),('right','/World/Replay_global_right',(640,480)),('wrist','/World/ReplayWrist',(640,360))]:
  product=rep.create.render_product(path,size);ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach([product]);products.append((name,ann))
 root=Path('/home/lp531/smolvla_eval/act_cup_only_frustum_steps15_paired30_20261025/batch_01');clips=[]
 for n,start,end in [(2,30,60),(5,30,60),(8,24,60),(12,24,32)]:
  ep=next(root.glob(f'*/episodes/cup_{n:03d}_*'));tr=[json.loads(l) for l in (ep/'trace.jsonl').read_text().splitlines()];selected=[]
  for sec in np.arange(start,end,.2):selected.append(min(tr,key=lambda t:abs(t['simulation_elapsed_s']-sec)))
  clips.append((n,selected))
 pose(clips[0][1][0]);rep.orchestrator.step(rt_subframes=16)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24);manifest=[];started=time.monotonic()
 for phase in ['before','after']:
  settings=carb.settings.get_settings()
  if phase=='before':settings.set('/rtx/translucency/worldEps',1.)
  else:configure_close_range_refraction(settings)
  for trial,rows in clips:
   pose(rows[0]);rep.orchestrator.step(rt_subframes=16);dest=R/f'{phase}_trial{trial:02d}.mp4';writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),5,(1920,516));assert writer.isOpened();checks=[]
   for i,t in enumerate(rows):
    pose(t);rep.orchestrator.step(rt_subframes=4);page=Image.new('RGB',(1920,516),'black');draw=ImageDraw.Draw(page)
    for j,(name,ann) in enumerate(products):
     im=ann.get_data();assert im.size;im=Image.fromarray(im[:,:,:3]);page.paste(im,(j*640,36+(480-im.height)//2))
    draw.text((10,5),f'{phase.upper()} | Trial {trial:02d} | sim={t["simulation_elapsed_s"]:.2f}s | recorded-pose replay | worldEps={settings.get("/rtx/translucency/worldEps")}',font=font,fill='white');writer.write(cv2.cvtColor(np.asarray(page),cv2.COLOR_RGB2BGR))
    if i%25==0 or i==len(rows)-1:page.save(R/f'{phase}_trial{trial:02d}_frame{i:03d}.jpg',quality=94);checks.append(i)
   writer.release();manifest.append(dict(phase=phase,trial=trial,frames=len(rows),fps=5,path=str(dest),review_frames=checks));print('CLIP_DONE',phase,trial,len(rows),flush=True)
 report=dict(clips=manifest,elapsed_s=time.monotonic()-started,mode='recorded pose rendering only; not a physics or ACT evaluation');(R/'manifest.json').write_text(json.dumps(report,indent=2));print('DYNAMIC_REPLAY_DONE',report['elapsed_s'],flush=True)
except Exception:
 import traceback;traceback.print_exc()
finally:app.close()
