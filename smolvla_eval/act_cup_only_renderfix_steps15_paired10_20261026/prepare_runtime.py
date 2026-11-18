from pathlib import Path
import ast,json,difflib
r=Path(__file__).parent
source=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py')
s=source.read_text();original=s
# Resolve shared helpers from their original location before local imports.
s=s.replace('import time\n','import time\nfrom pathlib import Path\nsys.path.insert(0, '+repr(str(source.parent))+')\nCUP_ONLY_POSES = json.loads(Path('+repr(str(r/'paired_cup_poses.json'))+').read_text())\n',1)
for name,replacement in [
 ('TRAINING_OBJECT_PRIMS',"TRAINING_OBJECT_PRIMS = ('/World/Task/Objects/Cup',)"),
 ('OBJECT_NAME_TO_PRIM',"OBJECT_NAME_TO_PRIM = {'cup': '/World/Task/Objects/Cup'}"),
 ('ACTIVE_OBJECT_COLLIDER_PATHS',"ACTIVE_OBJECT_COLLIDER_PATHS = {'cup': '/World/Task/Objects/Cup/Collider'}")]:
 tree=ast.parse(s)
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
 lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=[replacement+'\n'];s=''.join(lines)
s=s.replace('    for object_name in ("apple", "peach", "mouse"):\n','    for object_name in ("apple", "peach", "mouse"):\n        if object_name not in OBJECT_NAME_TO_PRIM:\n            continue\n',1)
s=s.replace('        self._target_object = "apple"\n', '        self._target_object = "cup"\n', 1)
s=s.replace('    if len(configured) < 10:\n','    if len(configured) < len(ACTIVE_OBJECT_COLLIDER_PATHS) + 8:\n',1)
start=s.index('        # The mouse\'s scanned mesh')
end=s.index('        self._context = ',start)
s=s[:start]+s[end:]
start=s.index('    def _sample_randomized_object_poses(self, seed, level="easy"):')
end=s.index('    def _apply_scene_randomization(',start)
s=s[:start]+'''    def _sample_randomized_object_poses(self, seed, level="easy"):
        # Replay the baseline cup pose, avoiding a changed randomization layout
        # when the non-target sampling/rejection steps no longer exist.
        if self._target_object != "cup":
            raise ValueError("This isolated evaluation scene only contains cup")
        pose = CUP_ONLY_POSES[str(int(seed))]
        return {"cup": (np.asarray(pose["position"], dtype=np.float32),
                        np.asarray(pose["orientation"], dtype=np.float32))}

'''+s[end:]
(r/'run_cup_only_scene.py').write_text(s)
(r/'runtime_changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(keepends=True),s.splitlines(keepends=True),fromfile=str(source),tofile=str(r/'run_cup_only_scene.py'))))
overlay=r/'ros_overlay/z1_isaac_bridge';overlay.mkdir(parents=True,exist_ok=True)
(overlay/'__init__.py').write_text('from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n')
(overlay/'object_collision_sync.py').write_text('''"""Experiment-only collision synchronization for the sole remaining object."""
import importlib.util
import sys
from pathlib import Path
source = Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/object_collision_sync.py')
spec = importlib.util.spec_from_file_location('z1_isaac_bridge._full_object_collision_sync', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.OBJECT_COLLISIONS = {'cup': module.OBJECT_COLLISIONS['cup']}
ObjectCollisionSync = module.ObjectCollisionSync
OBJECT_COLLISIONS = module.OBJECT_COLLISIONS
main = module.main
''')
p=r/'start.sh';s=p.read_text().replace('export ROS_DOMAIN_ID=98','export PYTHONPATH="'+str(r/'ros_overlay')+':${PYTHONPATH:-}"\nexport ROS_DOMAIN_ID=98');p.write_text(s)
p=r/'supervise.py';s=p.read_text().replace('generated_dual_camera/smolvla_dual_wrist_camera.usd','generated_dual_camera/smolvla_cup_only_frustum_20261025.usd');s=s.replace("'external_camera_layout:=dual'",repr('isaac_script:='+str(r/'run_cup_only_scene.py'))+",'external_camera_layout:=dual'");p.write_text(s)
