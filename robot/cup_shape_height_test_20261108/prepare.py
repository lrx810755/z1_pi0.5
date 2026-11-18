import json,hashlib,shutil
from pathlib import Path
from experiment_core import cases
R=Path(__file__).resolve().parent;B=Path('/home/lp531/robot/cup_horizontal_demo_20261108');S=Path('/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts')
L=Path('/home/lp531/z1_queue_logs/cup_shape_height_test_20261108');L.mkdir(exist_ok=True)
source_files=[B/n for n in ['assistant.py','base_assistant_snapshot.py','core.py','geometry.py','grasp_physics.json','move_group.launch.py','collection.launch.py']]+[S/'run_z1_ros2_scene.py']
(L/'original_source_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},indent=2))
for name in ['assistant.py','base_assistant_snapshot.py','core.py','geometry.py','move_group.launch.py','set_collection_column.py']:
 shutil.copy2(B/name,R/name)
config=json.loads((B/'grasp_physics.json').read_text())
for height in [34,32]:
 config['object_grasp_profiles']['cup']['grasp_height_target_m']=height/1000
 (R/f'grasp_h{height}.json').write_text(json.dumps(config,indent=2))
launch=(B/'collection.launch.py').read_text().replace(str(B),str(R))
launch=launch.replace('import json\n','import json\nimport os\n').replace("grasp_config = '"+str(R/ 'grasp_physics.json')+"'","grasp_config = os.environ['Z1_ISAAC_GRASP_CONFIG']")
launch=launch.replace('package="z1_isaac_bridge",\n                executable="object_collision_sync",','executable="/usr/bin/python3",\n                arguments=["'+str(R/'collision_sync.py')+'"],')
# Recorder stays idle during experiments. All videos are preserved by the probe.
launch=launch.replace('arguments=["'+str(R/'failure_recorder.py')+'"],','arguments=["'+str(B/'failure_recorder.py')+'"],')
(R/'collection.launch.py').write_text(launch)
scene=(S/'run_z1_ros2_scene.py').read_text()
scene=scene.replace('import time\n','import time\nfrom pathlib import Path\n',1)
scene=scene.replace('from layout_sweep import',f'sys.path.insert(0, {str(S)!r})\nfrom layout_sweep import',1)
anchor='    def _sample_randomized_object_poses(self, seed, level="easy"):\n'
assert scene.count(anchor)==1
scene=scene.replace(anchor,anchor+'''        case = json.loads(Path(os.environ['CUP_EXPERIMENT_CASE']).read_text())
        poses = {name: (np.array(self._initial_object_poses[path][0], copy=True), np.array(self._initial_object_poses[path][1], copy=True)) for name,path in OBJECT_NAME_TO_PRIM.items()}
        poses['cup'] = (np.asarray(case['cup_position_m'],dtype=np.float32), np.asarray([1.,0.,0.,0.],dtype=np.float32))
        return poses
''',1)
scene=scene.replace('    configure_fruit_collision_approximations(stage)\n','    configure_fruit_collision_approximations(stage)\n    from scene_geometry import configure_experiment_cup\n    configure_experiment_cup(stage)\n',1)
(R/'scene.py').write_text(scene)
(L/'cases.json').write_text(json.dumps(cases(),indent=2))
print('Prepared 120 paired cases and isolated production snapshots')
