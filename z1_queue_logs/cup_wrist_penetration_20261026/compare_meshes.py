from pxr import Usd,UsdGeom,UsdPhysics
import numpy as np,json
from pathlib import Path
s=Usd.Stage.Open('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd');x=UsdGeom.XformCache();out=[]
for name in ['gripperStator','gripperMover']:
 root=s.GetPrimAtPath('/World/Z1/root_joint/'+name);inv=x.GetLocalToWorldTransform(root).GetInverse();meshes={}
 for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
  if p.IsA(UsdGeom.Mesh):
   m=UsdGeom.Mesh(p);tf=x.GetLocalToWorldTransform(p)*inv;points=np.array([tuple(tf.Transform(tuple(v))) for v in m.GetPointsAttr().Get()]);role='collision' if '/collisions/' in str(p.GetPath()) else 'visual';meshes[role]=points
 a=meshes['visual'];b=meshes['collision'];same_order=float(np.max(np.linalg.norm(a-b,axis=1))) if a.shape==b.shape else None;out.append(dict(link=name,visual_vertices=len(a),collision_vertices=len(b),maximum_vertex_difference_m=same_order,visual_min=a.min(0).tolist(),collision_min=b.min(0).tolist(),visual_max=a.max(0).tolist(),collision_max=b.max(0).tolist()))
print(json.dumps(out,indent=2));Path('/home/lp531/z1_queue_logs/cup_wrist_penetration_20261026/mesh_comparison.json').write_text(json.dumps(out,indent=2))
