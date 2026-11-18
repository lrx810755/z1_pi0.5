from pathlib import Path
r=Path(__file__).parent;s=(r/'depth_probe.py').read_text();s=s.replace("R=G/'depth_probe'","R=G/os.environ.get('PROBE_OUT','multi_view')")
s=s.replace(" rp=rep.create.render_product('/World/ProbeCamera',(640,360));", " rp=rep.create.render_product('/World/ProbeCamera',(640,360));")
start=s.index(" rp=rep.create.render_product")
end=s.index('except Exception:')
s=s[:start]+''' import sys,carb
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
''' +s[end:]
(r/'multi_view.py').write_text(s)
