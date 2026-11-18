"""Run the original PI0.5 scene with two session-only collision exclusions."""
import importlib.util
import json
from pathlib import Path
import sys

JOB = Path(__file__).resolve().parent
SOURCE = Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py')
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location('z1_original_scene', SOURCE)
scene = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scene
spec.loader.exec_module(scene)

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

FINGERS = (scene.GRIPPER_STATOR, scene.GRIPPER_MOVER)
BOX = '/World/Task/Box'
BOTTOM = BOX + '/Bottom'
OBJECTS = ('apple', 'peach', 'cup', 'beverage_bottle')
geometry = {}
collisions = {}


def filter_pair(stage):
    table = stage.GetPrimAtPath(scene.WORK_SURFACE)
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'proxy', 'render'])
    bounds = bbox.ComputeWorldBound(table).ComputeAlignedBox()
    geometry['min'] = np.array(bounds.GetMin())
    geometry['max'] = np.array(bounds.GetMax())
    if abs(geometry['max'][2] - scene.TABLE_CONFIG['top_z_m']) > .003:
        raise RuntimeError(f'Unexpected tabletop height {geometry["max"][2]}')
    required = [scene.WORK_SURFACE, *(scene.ACTIVE_OBJECT_COLLIDER_PATHS[o] for o in OBJECTS),
                *(BOX + '/' + name for name in ('Bottom', 'WallLeft', 'WallRight', 'WallFront', 'WallBack'))]
    for path in required:
        api = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path))
        if not api or not api.GetCollisionEnabledAttr().Get():
            raise RuntimeError(f'Expected enabled collision at {path}')
    filter_rel = UsdPhysics.FilteredPairsAPI.Apply(table).CreateFilteredPairsRel()
    prior = set(filter_rel.GetTargets())
    for link in FINGERS:
        prim = stage.GetPrimAtPath(link)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f'Expected finger rigid body at {link}')
        points = []
        local = cache.GetLocalToWorldTransform(prim).GetInverse()
        root = stage.GetPrimAtPath(link + '/collisions')
        for mesh in Usd.PrimRange(root):
            if mesh.IsA(UsdGeom.Mesh):
                transform = cache.GetLocalToWorldTransform(mesh) * local
                points.extend(tuple(transform.Transform(Gf.Vec3d(*x))) for x in UsdGeom.Mesh(mesh).GetPointsAttr().Get())
        if not points:
            raise RuntimeError(f'No finger collider geometry at {link}')
        geometry[link] = np.asarray(points)
        filter_rel.AddTarget(prim.GetPath())
    retained = {str(p) for p in filter_rel.GetTargets()}
    if retained != {str(p) for p in prior}.union(FINGERS):
        raise RuntimeError('Unexpected collision-pair changes')
    collisions.update(filtered_pairs=[[scene.WORK_SURFACE, link] for link in FINGERS],
                      original_targets=sorted(map(str, prior)), kept_enabled=required,
                      tabletop_z_m=float(geometry['max'][2]),
                      finger_vertices={link:len(geometry[link]) for link in FINGERS})
    print(json.dumps(collisions), flush=True)


original_configure = scene.configure_grasp_contact_reports


def configure(stage):
    original_configure(stage)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        filter_pair(stage)


scene.configure_grasp_contact_reports = configure


class FilteredScene(scene.SceneResetInterface):
    def __init__(self, robot, objects, gripper_path):
        super().__init__(robot, objects, gripper_path)
        self.support_test()

    def support_test(self):
        context = scene.SimulationContext.instance()
        zero = np.zeros(3, dtype=np.float32)
        stage = scene.omni.usd.get_context().get_stage()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'proxy', 'render'])
        bottom = bbox.ComputeWorldBound(stage.GetPrimAtPath(BOTTOM)).ComputeAlignedBox()
        center = np.array(bottom.GetMidpoint())
        drop_height = float(bottom.GetMax()[2]) + .12
        snapshots = {path: tuple(np.array(v).copy() for v in body.get_world_pose()) for path, body in self._objects.items()}

        def settle(seconds):
            for _ in range(round(seconds / context.get_physics_dt())):
                context.step(render=False)
                self.update_contact_state()

        def reposition(path, position):
            body = self._objects[path]
            body.set_world_pose(position=np.asarray(position), orientation=snapshots[path][1])
            body.set_linear_velocity(zero)
            body.set_angular_velocity(zero)

        support = {}
        try:
            settle(1.0)
            for obj in OBJECTS:
                path = scene.OBJECT_NAME_TO_PRIM[obj]
                body = self._objects[path]
                table_position = body.get_world_pose()[0]
                if obj not in self._object_table_contacts or abs(float(table_position[2] - snapshots[path][0][2])) > .02:
                    raise RuntimeError(f'Table support did not verify for {obj}: position={table_position}')
                reposition(path, [center[0], center[1], drop_height])
                settle(1.6)
                first_position = np.array(body.get_world_pose()[0])
                settle(.55)
                box_position = np.array(body.get_world_pose()[0])
                box_contact = any(
                    self._endpoint_is(pair['endpoint0'], BOTTOM) and self._endpoint_is(pair['endpoint1'], path)
                    or self._endpoint_is(pair['endpoint1'], BOTTOM) and self._endpoint_is(pair['endpoint0'], path)
                    for pair in self._active_contact_pairs.values()
                )
                drift = float(np.linalg.norm(box_position - first_position))
                if not box_contact or drift > .004 or not all(bottom.GetMin()[i] <= box_position[i] <= bottom.GetMax()[i] for i in (0,1)):
                    raise RuntimeError(f'Blue box support did not verify for {obj}: contact={box_contact}, drift={drift}, position={box_position}')
                support[obj] = dict(table_position_m=table_position.tolist(), table_contact=True,
                                    box_position_m=box_position.tolist(), box_bottom_contact=True,
                                    box_drift_after_half_second_m=drift)
                reposition(path, snapshots[path][0])
                settle(.4)
        finally:
            for path, (position, _) in snapshots.items():
                reposition(path, position)
            settle(.5)
        report = dict(passed=True, **collisions, support=support, geometry_note='finger collider mesh vertices within the tabletop XY footprint')
        temp = JOB / 'physics_verified.json.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(JOB / 'physics_verified.json')
        print('Verified actual table and box-bottom support for all four objects', flush=True)

    def _publish_grasp_status(self, phase, message, **details):
        if phase == 'CONTACT_STATE':
            stage = scene.omni.usd.get_context().get_stage()
            cache = UsdGeom.XformCache()
            depths = []
            lo, hi = geometry['min'], geometry['max']
            for link in FINGERS:
                transform = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(link)))
                xyz = geometry[link] @ transform[:3, :3] + transform[3, :3]
                above = np.all((xyz[:, :2] >= lo[:2]) & (xyz[:, :2] <= hi[:2]), axis=1)
                depths.append(float(max(0.0, hi[2] - xyz[above, 2].min())) if above.any() else 0.0)
            details['finger_table_penetration_estimate_m'] = depths
            details['finger_table_filter_active'] = True
        super()._publish_grasp_status(phase, message, **details)


scene.SceneResetInterface = FilteredScene
try:
    scene.main()
except Exception:
    import traceback
    traceback.print_exc()
    scene.simulation_app.close()
    raise
