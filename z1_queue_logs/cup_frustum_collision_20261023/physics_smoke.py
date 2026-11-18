"""Cook the actual cup collider in PhysX and probe contact at three heights."""
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'disable_viewport_updates':True})
import sys,json,math
from pathlib import Path
import numpy as np
from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
root=Path('/home/lp531/ros2_ws/src/z1_ros2')
sys.path.insert(0,str(root/'isaac_assets/z1/scripts'))
from cup_collision import CUP_COLLISION
out=Path(__file__).parent
try:
 world=World(physics_dt=1/120,rendering_dt=1/60,backend='numpy',device='cpu')
 stage=world.stage
 scene=world.get_physics_context()._physics_scene
 scene.CreateGravityDirectionAttr(Gf.Vec3f(-1,0,0))
 scene.CreateGravityMagnitudeAttr(9.81)
 cup=UsdGeom.Xform.Define(stage,'/World/Cup').GetPrim()
 cup.GetReferences().AddReference(str(root/'isaac_assets/z1/generated/smolvla_red_ball_into_box.usd'),'/World/Task/Objects/Cup')
 UsdGeom.Xformable(cup).ClearXformOpOrder()
 UsdPhysics.RigidBodyAPI(cup).CreateKinematicEnabledAttr(True)
 PhysxSchema.PhysxRigidBodyAPI(cup).CreateEnableCCDAttr(False)
 assert stage.GetPrimAtPath('/World/Cup/Collider').IsA(UsdGeom.Mesh)
 mat=UsdShade.Material.Define(stage,'/World/Looks/ObjectPhysics')
 physmat=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
 physmat.CreateStaticFrictionAttr(0);physmat.CreateDynamicFrictionAttr(0);physmat.CreateRestitutionAttr(0)
 UsdShade.Material.Define(stage,'/World/Looks/CollisionPreview')
 col=stage.GetPrimAtPath('/World/Cup/Collider')
 UsdShade.MaterialBindingAPI.Apply(col).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
 col.CreateRelationship('material:binding').SetTargets([])
 probes=[]
 for i,z in enumerate([.006,.034,.056]):
  path=f'/World/Probe{i}'
  sphere=UsdGeom.Sphere.Define(stage,path);sphere.CreateRadiusAttr(.003)
  UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(.06,0,z))
  p=sphere.GetPrim();UsdPhysics.RigidBodyAPI.Apply(p);UsdPhysics.MassAPI.Apply(p).CreateMassAttr(.01)
  UsdPhysics.CollisionAPI.Apply(p)
  PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateLockedPosAxisAttr(6)
  PhysxSchema.PhysxRigidBodyAPI(p).CreateLockedRotAxisAttr(7)
  UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
  probes.append((z,world.scene.add(SingleRigidPrim(prim_path=path,name=f'probe{i}'))))
 for path in ['/World/Cup/Collider']+[f'/World/Probe{i}' for i in range(3)]:
  api=PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(path));api.CreateContactOffsetAttr(.00025);api.CreateRestOffsetAttr(0)
 world.reset()
 for _ in range(300):world.step(render=False)
 results=[]
 slope=(.0252-.018)/.061
 for z,body in probes:
  pos,quat=body.get_world_pose()
  expected=.018+slope*z+.003*math.sqrt(1+slope*slope)
  error=float(pos[0])-expected
  results.append({'height_m':z,'sphere_center_m':pos.tolist(),'expected_contact_x_m':expected,'error_m':error})
  assert abs(error)<.0004,(z,pos,expected)
  assert abs(float(pos[2])-z)<.0001
 assert results[0]['sphere_center_m'][0]<results[1]['sphere_center_m'][0]<results[2]['sphere_center_m'][0]
 report={'verified':True,'physics':'PhysX CPU 120 Hz','source':'active generated scene cup reference','contact_tests':results,'full_robot_grasp_tested':False}
 (out/'physics_verified.json').write_text(json.dumps(report,indent=2)+'\n')
 print('PHYSICS_VERIFIED',json.dumps(report),flush=True)
finally:
 app.close()
