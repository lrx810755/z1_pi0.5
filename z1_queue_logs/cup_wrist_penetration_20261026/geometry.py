from pxr import Usd,UsdGeom,UsdShade,Gf
import numpy as np,json
from pathlib import Path
R=Path(__file__).parent
s=Usd.Stage.Open('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_cup_only_frustum_20261025.usd');xc=UsdGeom.XformCache()
st='/World/Z1/root_joint/gripperStator';cup='/World/Task/Objects/Cup';arrays={};rows={}
for prefix,root in [('stator',st),('mover','/World/Z1/root_joint/gripperMover'),('cup',cup)]:
 inv=xc.GetLocalToWorldTransform(s.GetPrimAtPath(root)).GetInverse()
 for p in Usd.PrimRange(s.GetPrimAtPath(root),Usd.TraverseInstanceProxies()):
  if not p.IsA(UsdGeom.Mesh) or ('/collisions/' in str(p.GetPath())):continue
  m=UsdGeom.Mesh(p);tf=xc.GetLocalToWorldTransform(p)*inv
  pts=np.array([list(tf.Transform(Gf.Vec3d(*v))) for v in m.GetPointsAttr().Get()]);key=prefix+'_'+str(len(arrays));arrays[key]=pts
  rows[key]=dict(path=str(p.GetPath()),min=pts.min(0).tolist(),max=pts.max(0).tolist())
  if prefix=='cup':
   rr=np.linalg.norm(pts[:,:2],axis=1);ideal=.018+pts[:,2]/.061*.0072;ex=rr-ideal
   rows[key]['max_radial_overhang_mm']=float(ex.max()*1000)
   rows[key]['points_outside_0p5mm']=int((ex>.0005).sum())
  if prefix=='cup' and 'Object_2' in str(p.GetPath()):
   inds=list(m.GetFaceVertexIndicesAttr().Get());faces=[];cursor=0
   for count in m.GetFaceVertexCountsAttr().Get():
    face=inds[cursor:cursor+count];cursor+=count
    for i in range(1,count-1):faces.append([face[0],face[i],face[i+1]])
   arrays['cup_faces']=np.array(faces)
  if prefix=='mover':
   counts=m.GetFaceVertexCountsAttr().Get();inds=np.array(m.GetFaceVertexIndicesAttr().Get());assert all(n==3 for n in counts);arrays['mover_faces']=inds.reshape(-1,3)
  if prefix=='stator':
   counts=m.GetFaceVertexCountsAttr().Get();inds=np.array(m.GetFaceVertexIndicesAttr().Get());assert all(n==3 for n in counts);arrays['stator_faces']=inds.reshape(-1,3)
# actual runtime camera transform
rot=Gf.Rotation(Gf.Vec3d(1,0,0),-90)*Gf.Rotation(Gf.Vec3d(0,0,1),-90)*Gf.Rotation(Gf.Vec3d(1,0,0),-5)*Gf.Rotation(Gf.Vec3d(0,1,0),15)
cm=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),180))*Gf.Matrix4d().SetRotate(rot)
cm.SetTranslateOnly(Gf.Vec3d(.02,-.08,.04));ci=cm.GetInverse();rows['camera']={'position_stator':[.02,-.08,.04],'matrix_row_vector':list(map(list,cm)),'near_clip_m':.02}
pts=arrays['stator_0'];cp=np.array([list(ci.Transform(Gf.Vec3d(*v))) for v in pts]);depth=-cp[:,2]
rows['camera']['stator_min_euclidean_distance_to_vertices_m']=float(np.linalg.norm(pts-np.array([.02,-.08,.04]),axis=1).min())
rows['camera']['stator_vertex_depth_range_m']=[float(depth.min()),float(depth.max())]
rows['camera']['stator_vertices_in_front_inside_near_clip']=int(((depth>0)&(depth<.02)).sum())
for path in ['/World/Z1/root_joint/Looks/material_Grey/Shader',cup+'/VisualAsset/Normalized/Visual/chai_glass_obj/Materials/Glass/pbr_shader',cup+'/VisualAsset/Normalized/Visual/chai_glass_obj/Materials/Condensation/pbr_shader']:
 p=s.GetPrimAtPath(path);rows[path]={a.GetName():str(a.Get()) for a in p.GetAttributes()}
np.savez(R/'geometry.npz',**arrays);(R/'geometry.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
