import os,sys,json,copy,datetime,pathlib,hashlib,importlib.util
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import ListParameters,GetParameters
from rclpy.parameter import parameter_value_to_python
ROOT=pathlib.Path('/home/lp531/robot/apple_scale110_correction60_20261106');OUT=pathlib.Path(__file__).resolve().parent
os.environ['Z1_ISAAC_GRASP_CONFIG']=str(ROOT/'grasp_physics.json');sys.path.insert(0,str(ROOT))
import z1_isaac_bridge.grasp_assistant as ga
original=copy.deepcopy(ga.GRASP_PROFILES['apple']);ga.main=lambda:None
spec=importlib.util.spec_from_file_location('active_apple_nodes',ROOT/'nodes.py');private=importlib.util.module_from_spec(spec);spec.loader.exec_module(private);private.assistant_main()
physics=json.loads((ROOT/'grasp_physics.json').read_text())
relevant={k:physics[k] for k in ['gripper','physical_test_lift','table','unfreeze','scene_randomization','contact_report','grasp_envelope']}
relevant['apple_collision_proxy']=physics['collision_proxy']['apple'];relevant['fruit_contact_proxy_common']={k:v for k,v in physics['collision_proxy'].items() if k not in ['apple','peach','mouse']};relevant['apple_center_z_offset_m']=physics['object_center_z_offsets_m']['apple']
constants={}
for k,v in vars(ga).items():
 if k.isupper():
  if isinstance(v,dict) and 'apple' in v:constants[k]=v['apple']
  elif isinstance(v,(int,float,bool,str,list,tuple)):
   try:json.dumps(v);constants[k]=v
   except TypeError:pass
rclpy.init();node=Node('apple_parameter_readonly_export');runtime={}
try:
 for name in ['/smolvla_grasp_assistant','/smolvla_grasp_evaluator','/smolvla_episode_recorder']:
  l=node.create_client(ListParameters,name+'/list_parameters');g=node.create_client(GetParameters,name+'/get_parameters')
  if not l.wait_for_service(timeout_sec=5):runtime[name]={'error':'service unavailable'};continue
  f=l.call_async(ListParameters.Request());rclpy.spin_until_future_complete(node,f,timeout_sec=10)
  if not f.done():raise RuntimeError('list timeout '+name)
  names=f.result().result.names;f=g.call_async(GetParameters.Request(names=names));rclpy.spin_until_future_complete(node,f,timeout_sec=10)
  if not f.done():raise RuntimeError('get timeout '+name)
  runtime[name]={k:parameter_value_to_python(v) for k,v in zip(names,f.result().values)}
finally:node.destroy_node();rclpy.shutdown()
plan=json.loads(pathlib.Path('/home/lp531/smolvla_data/apple_scale110_correction60_20261106/correction_plan.json').read_text())
files=[ROOT/'nodes.py',ROOT/'fragment_core.py',ROOT/'wait_optimization.py',ROOT/'grasp_physics.json',ROOT/'preflight.py',pathlib.Path(ga.__file__).resolve()]
result=dict(exported_at=datetime.datetime.now().astimezone().isoformat(),active_process_pid=3860187,active_source_directory=str(ROOT),production_assistant_source=str(pathlib.Path(ga.__file__).resolve()),apple_profile_before_private_overrides=original,apple_profile_effective=ga.GRASP_PROFILES['apple'],private_overrides={k:v for k,v in ga.GRASP_PROFILES['apple'].items() if k not in original or original[k]!=v},shared_physics=relevant,apple_related_constants=constants,live_ros_parameters=runtime,correction_plan=plan,source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
(OUT/'苹果抓取全部参数.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
for name,values in runtime.items():print(name,len(values),'parameters')
print('APPLE PROFILE',json.dumps(ga.GRASP_PROFILES['apple'],ensure_ascii=False,indent=2));print('PRIVATE OVERRIDES',result['private_overrides'])
