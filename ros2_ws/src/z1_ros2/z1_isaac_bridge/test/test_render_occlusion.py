"""Verify the tabletop refraction setting survives a USD save/reopen."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[2] / "isaac_assets/z1/scripts"


class RenderOcclusionTests(unittest.TestCase):
    def test_scene_setting_survives_reopen_without_changing_scene_content(self):
        try:
            from pxr import Sdf, Usd, UsdGeom, UsdShade
        except ImportError:
            self.skipTest("Run USD integration with Isaac's USD Python")

        spec = importlib.util.spec_from_file_location(
            "render_occlusion", SCRIPTS / "render_occlusion.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        author_setting = getattr(module, "author_close_range_refraction", None)
        self.assertTrue(callable(author_setting), "Scene files cannot persist the refraction fix")

        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "tabletop.usda")
            stage = Usd.Stage.CreateNew(path)
            UsdGeom.Cube.Define(stage, "/World/Box")
            shader = UsdShade.Shader.Define(stage, "/World/Bottle/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.25)
            layer = stage.GetRootLayer()
            layer.customLayerData = {
                "taskName": "multi_object_tabletop_manipulation",
                "renderSettings": {"rtx:translucency:maxRefractionBounces": 8},
            }
            content = Sdf.Layer.CreateAnonymous()
            content.TransferContent(layer)
            content.customLayerData = {}
            before = content.ExportToString()

            author_setting(stage)
            once = layer.ExportToString()
            author_setting(stage)
            self.assertEqual(layer.ExportToString(), once, "Repeated authoring changed the scene")
            layer.Save()
            stage = None
            layer = None

            reopened = Usd.Stage.Open(path)
            saved = reopened.GetRootLayer()
            self.assertEqual(saved.customLayerData, {
                "taskName": "multi_object_tabletop_manipulation",
                "renderSettings": {
                    "rtx:translucency:maxRefractionBounces": 8,
                    "rtx:translucency:worldEps": 0.0001,
                },
            })
            content.TransferContent(saved)
            content.customLayerData = {}
            self.assertEqual(content.ExportToString(), before)


if __name__ == "__main__":
    unittest.main()
