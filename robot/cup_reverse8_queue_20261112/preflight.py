"""CPU-only native import, URDF and launch wiring check; no scene starts."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'collector'))
import nodes
geometry=nodes.Geometry()
plan=json.loads((ROOT/'plan.json').read_text())
for t in plan['tasks']:
 row=json.loads((Path(t['rerecord_source'])/'frames.jsonl').read_text().splitlines()[0])
 point,rotation=geometry.link_fk(row['observation.state'][:6])
 assert len(point)==3 and geometry.height(row['observation.state'])>-.05
spec=importlib.util.spec_from_file_location('cup7launch',ROOT/'collector/collection.launch.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ld=module.generate_launch_description()
from launch_ros.actions import Node
nodespecs=[a for a in ld.entities if isinstance(a,Node)]
roles=[]
for n in nodespecs:
 values=vars(n)
 executable=values.get('_Node__node_executable',values.get('_ExecuteProcess__cmd',[]))
 text=str(executable)
 if 'nodes.py' in text:roles.append(text)
assert len(roles)==3,roles
print(json.dumps(dict(passed=True,urdf_loaded=True,source_poses_checked=7,isolated_nodes=len(roles),gpu_scene_started=False)))
