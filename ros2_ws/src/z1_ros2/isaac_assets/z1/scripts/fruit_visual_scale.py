"""Scale a fruit visual independently of its separately configured collider."""
import math

def scale_fruit_visual(stage, root_path, factor):
    factor=float(factor)
    if not math.isfinite(factor) or factor<=0:
        raise ValueError('Fruit visual scale must be finite and positive')
    if factor==1.:
        return
    from pxr import UsdGeom, Gf
    prim=stage.GetPrimAtPath(root_path+'/VisualAsset')
    if not prim.IsValid():
        raise RuntimeError('Fruit visual missing: '+root_path)
    xform=UsdGeom.Xformable(prim)
    ops=xform.GetOrderedXformOps()
    existing=[op for op in ops if op.GetOpName()=='xformOp:scale:demoSize']
    op=existing[0] if existing else xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble,opSuffix='demoSize')
    op.Set(Gf.Vec3d(factor,factor,factor))
    xform.SetXformOpOrder([op]+[old for old in ops if old.GetOpName()!='xformOp:scale:demoSize'])
