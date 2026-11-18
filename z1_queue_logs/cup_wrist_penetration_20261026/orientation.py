from pxr import Usd,UsdGeom,UsdShade
import json
s=Usd.Stage.Open('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd');xc=UsdGeom.XformCache()
for p in Usd.PrimRange.Stage(s,Usd.TraverseInstanceProxies()):
 if '/visuals/' not in str(p.GetPath()) or not p.IsA(UsdGeom.Mesh):continue
 m=UsdGeom.Mesh(p)
 print(p.GetPath(),'orientation',m.GetOrientationAttr().Get(),'double',m.GetDoubleSidedAttr().Get(),'det',xc.GetLocalToWorldTransform(p).GetDeterminant(),'normals',len(m.GetNormalsAttr().Get() or []),'normalInterp',m.GetNormalsInterpolation())
