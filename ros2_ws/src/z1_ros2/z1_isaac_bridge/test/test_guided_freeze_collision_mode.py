import ast,json
from pathlib import Path
from types import SimpleNamespace
SOURCE=Path(__file__).parents[1]/'z1_isaac_bridge'/'grasp_assistant.py'
def test_guided_freeze_can_preserve_real_collisions():
 tree=ast.parse(SOURCE.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(x,ast.FunctionDef) and x.name=='_command_scene_grasp' for x in n.body)); method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_command_scene_grasp')
 namespace={'json':json,'String':lambda:SimpleNamespace(data=''),'time':SimpleNamespace(sleep=lambda _:None)}
 exec(compile(ast.Module(body=[method],type_ignores=[]),str(SOURCE),'exec'),namespace)
 sent=[]
 class Lock:
  def __enter__(self):pass
  def __exit__(self,*args):pass
 node=SimpleNamespace(_lock=Lock(),_scene_grasp_pub=SimpleNamespace(publish=lambda msg:sent.append(json.loads(msg.data))),get_parameter=lambda key:SimpleNamespace(value=True))
 namespace['_command_scene_grasp'](node,'freeze','beverage_bottle')
 assert len(sent)==3
 assert all(payload.get('keep_collisions') is True for payload in sent)

def test_contact_telemetry_does_not_replace_command_acknowledgement():
 tree=ast.parse(SOURCE.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(x,ast.FunctionDef) and x.name=='_scene_grasp_status_callback' for x in n.body)); method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_scene_grasp_status_callback')
 namespace={'json':json,'String':object};exec(compile(ast.Module(body=[method],type_ignores=[]),str(SOURCE),'exec'),namespace)
 class Lock:
  def __enter__(self):pass
  def __exit__(self,*args):pass
 node=SimpleNamespace(_lock=Lock(),_scene_grasp_status={'phase':'FROZEN','target_collisions_enabled':True})
 namespace['_scene_grasp_status_callback'](node,SimpleNamespace(data=json.dumps({'phase':'CONTACT_STATE','bilateral_contact':True})))
 assert node._scene_grasp_status['phase']=='FROZEN'
 assert node._scene_grasp_contact_status['bilateral_contact'] is True
