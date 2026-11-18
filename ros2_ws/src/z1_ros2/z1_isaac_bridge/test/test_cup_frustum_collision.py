"""Regress the physical cup silhouette, solid topology, and USD authoring."""
import importlib.util
import math
from pathlib import Path
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / 'isaac_assets/z1/scripts'


class CupFrustumCollisionTests(unittest.TestCase):
    def setUp(self):
        path = SCRIPTS / 'cup_collision.py'
        self.assertTrue(path.exists(), 'Cup still uses the cylindrical collision model')
        spec = importlib.util.spec_from_file_location('cup_collision', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_silhouette_matches_base_rim_and_grasp_height(self):
        points, counts, indices = self.module.cup_frustum_mesh()
        bottom = [p for p in points if abs(p[2]) < 1e-12]
        top = [p for p in points if abs(p[2] - .061) < 1e-12]
        self.assertEqual(len(bottom), 64)
        self.assertEqual(len(top), 64)
        for p in bottom:
            self.assertAlmostEqual(math.hypot(*p[:2]), .018)
        for p in top:
            self.assertAlmostEqual(math.hypot(*p[:2]), .0252)
        radius_at_grasp = .018 + (.0252 - .018) * .034 / .061
        self.assertAlmostEqual(radius_at_grasp, .02201311475409836)

    def test_closed_outward_faces_enclose_correct_volume(self):
        points, counts, indices = self.module.cup_frustum_mesh()
        edges = {}
        volume = 0.0
        start = 0
        for count in counts:
            face = indices[start:start + count]
            start += count
            for a, b in zip(face, face[1:] + face[:1]):
                edges[(a, b)] = edges.get((a, b), 0) + 1
            p = points[face[0]]
            for j in range(1, len(face) - 1):
                q, r = points[face[j]], points[face[j + 1]]
                cross = (q[1]*r[2]-q[2]*r[1], q[2]*r[0]-q[0]*r[2], q[0]*r[1]-q[1]*r[0])
                volume += sum(p[k]*cross[k] for k in range(3)) / 6
        self.assertEqual(start, len(indices))
        for (a, b), count in edges.items():
            self.assertEqual(count, 1)
            self.assertEqual(edges.get((b, a)), 1, 'Mesh has an open edge or inconsistent winding')
        # Exact volume of the regular 64-sided frustum, with positive outward winding.
        expected = 64 * math.sin(2*math.pi/64) / 2 * .061 / 3 * (.018**2 + .018*.0252 + .0252**2)
        self.assertAlmostEqual(volume, expected, places=12)

    def test_usd_collider_is_convex_invisible_and_uses_original_material(self):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
        except ImportError:
            self.skipTest('USD integration is run separately with Isaac USD Python')
        stage = Usd.Stage.CreateInMemory()
        material = UsdShade.Material.Define(stage, '/PhysicsMaterial')
        prim = self.module.define_cup_collider(stage, '/Cup/Collider', material)
        self.assertTrue(prim.IsA(UsdGeom.Cylinder))
        cylinder=UsdGeom.Cylinder(prim)
        self.assertAlmostEqual(cylinder.GetRadiusAttr().Get(), .0252)
        self.assertAlmostEqual(cylinder.GetHeightAttr().Get(), .061)
        self.assertAlmostEqual(UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()[2], .0305)
        self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(UsdGeom.Imageable(prim).GetVisibilityAttr().Get(), 'invisible')
        self.assertEqual(UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel('physics').GetTargets(), [material.GetPath()])
        extent = UsdGeom.Cylinder(prim).GetExtentAttr().Get()
        self.assertAlmostEqual(extent[0][2], -.0305, places=7)
        self.assertAlmostEqual(extent[1][2], .0305, places=7)


if __name__ == '__main__':
    unittest.main()
