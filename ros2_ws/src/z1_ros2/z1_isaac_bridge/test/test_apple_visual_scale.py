import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import Mock
import sys
import pytest

PATH=Path(__file__).resolve().parents[2]/'isaac_assets/z1/scripts/fruit_visual_scale.py'
def test_scales_entire_visual_transform_around_object_base(monkeypatch):
    spec=importlib.util.spec_from_file_location('fruit_visual_scale',PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    translate,old_scale,new_scale=Mock(),Mock(),Mock()
    translate.GetOpName.return_value='xformOp:translate';old_scale.GetOpName.return_value='xformOp:scale'
    xform=Mock();xform.GetOrderedXformOps.return_value=[translate,old_scale];xform.AddScaleOp.return_value=new_scale
    monkeypatch.setitem(sys.modules,'pxr',N(UsdGeom=N(Xformable=Mock(return_value=xform),XformOp=N(PrecisionDouble='double')),Gf=N(Vec3d=lambda *x:x)))
    prim=Mock();prim.IsValid.return_value=True;stage=Mock();stage.GetPrimAtPath.return_value=prim
    module.scale_fruit_visual(stage,'/World/Task/Objects/Apple',1.1)
    stage.GetPrimAtPath.assert_called_once_with('/World/Task/Objects/Apple/VisualAsset')
    old_scale.Set.assert_not_called();translate.Set.assert_not_called()
    new_scale.Set.assert_called_once_with((1.1,1.1,1.1))
    xform.SetXformOpOrder.assert_called_once_with([new_scale,translate,old_scale])
