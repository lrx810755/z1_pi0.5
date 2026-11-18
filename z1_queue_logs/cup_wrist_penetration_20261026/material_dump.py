from pxr import Usd,UsdShade
s=Usd.Stage.Open('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd')
for p in s.Traverse():
 if '/Cup/' in str(p.GetPath()) and (p.IsA(UsdShade.Shader) or p.IsA(UsdShade.Material)):
  print(p.GetPath())
  for a in p.GetAttributes():print(' ',a.GetName(),str(a.Get()),[str(c) for c in a.GetConnections()])
