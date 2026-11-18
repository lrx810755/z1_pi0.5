from pathlib import Path
R=Path(__file__).resolve().parent;L=Path('/home/lp531/z1_queue_logs/cup_shape_height_test_20261108')
p=R/'assistant.py';s=p.read_text().replace("Path('/home/lp531/z1_queue_logs/cup_horizontal_demo_20261108')","Path(__import__('os').environ['CUP_EXPERIMENT_GROUP_DIR'])");p.write_text(s)
s=Path('/home/lp531/z1_queue_logs/cup_frustum_grasp_validation_20261026/run_cases.py').read_text()
s=s.replace('R=Path(__file__).parent',f'R=Path({str(L)!r})\nfrom experiment_core import passed')
s=s.replace('127.0.0.1:8810','127.0.0.1:8838').replace("=='94'","=='135'")
s=s.replace("atomic(R/'current_case.json',case);","atomic(R/'current_case.json',case);")
s=s.replace("  time.sleep(.5);node.start(d)","""  time.sleep(.5)
  node.set_column(case['target_grid'][0])
  position=node.status.get('object_position_m')
  if position is None or np.linalg.norm(np.array(position[:2])-case['cup_position_m'][:2])>.002:
   raise RuntimeError('Fixed cup pose verification failed: '+str(position))
  atomic(d/'initial_status.json',dict(status=node.status,scene=node.scene))
  node.start(d)""")
s=s.replace('def call(self,name,timeout=180):','def call(self,name,timeout=900):')
s=s.replace(" result['passed']=bool(result.get('service_success') and ever and last.get('physical_grasp_ok') and maximum-initial>=.03 and not last.get('object_touching_table',True))"," result['passed']=passed(result)")
s=s.replace("  for case in cases:results.append(run(node,case))", """  node.verify_geometry()
  for case in cases:
   existing=R/'cases'/case['id']/'result.json'
   if existing.exists():results.append(json.loads(existing.read_text()));continue
   results.append(run(node,case))
   atomic(R/'progress.json',dict(completed=len(list((R/'cases').glob('*/result.json'))),total=120,last=results[-1]))""")
# Parameter setter via direct services avoids ROS CLI discovery hangs.
anchor=' def call(self,name,timeout=900):'
extra=''' def wait_future(self,f,timeout=10):
  deadline=time.monotonic()+timeout
  while not f.done() and time.monotonic()<deadline:time.sleep(.05)
  if not f.done():raise RuntimeError('Bounded ROS request timed out')
  return f.result()
 def set_column(self,column):
  from rcl_interfaces.srv import SetParameters,GetParameters
  from rclpy.parameter import Parameter
  c=self.create_client(SetParameters,'/smolvla_grasp_assistant/set_parameters')
  try:
   if not c.wait_for_service(timeout_sec=5):raise RuntimeError('Column setter unavailable')
   req=SetParameters.Request();req.parameters=[Parameter('cup_collection_column',value=column).to_parameter_msg()]
   answer=self.wait_future(c.call_async(req));assert answer.results[0].successful,answer.results[0].reason
  finally:self.destroy_client(c)
  c=self.create_client(GetParameters,'/smolvla_grasp_assistant/get_parameters')
  try:
   assert c.wait_for_service(timeout_sec=5)
   req=GetParameters.Request();req.names=['cup_collection_column'];assert self.wait_future(c.call_async(req)).values[0].integer_value==column
  finally:self.destroy_client(c)
 def verify_geometry(self):
  from moveit_msgs.srv import GetPlanningScene
  from moveit_msgs.msg import PlanningSceneComponents
  c=self.create_client(GetPlanningScene,'/get_planning_scene')
  try:
   assert c.wait_for_service(timeout_sec=30),'Missing actual MoveIt scene service'
   req=GetPlanningScene.Request();req.components.components=PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
   answer=self.wait_future(c.call_async(req),30)
   objects=[o for o in answer.scene.world.collision_objects if o.id=='smolvla_object_cup']
   assert len(objects)==1,'Cup collider missing in actual MoveIt scene'
   obj=objects[0];shape=os.environ['CUP_EXPERIMENT_SHAPE']
   if shape=='frustum':
    assert len(obj.meshes)==1 and not obj.primitives
    vertices=np.array([[v.x,v.y,v.z] for v in obj.meshes[0].vertices]);assert len(vertices)==128
    assert abs(vertices[:,2].max()-.061)<1e-6
    for z,radius in [(0.,.018),(.061,.0252)]:assert np.allclose(np.linalg.norm(vertices[np.isclose(vertices[:,2],z),:2],axis=1),radius,atol=1e-6)
   else:
    assert len(obj.primitives)==1 and not obj.meshes
    assert obj.primitives[0].type==obj.primitives[0].CYLINDER
    assert np.allclose(obj.primitives[0].dimensions,[.061,.0252],atol=1e-8)
   atomic(Path(os.environ['CUP_EXPERIMENT_GROUP_DIR'])/'moveit_geometry_verified.json',message_to_ordereddict(obj))
   physical=json.loads((Path(os.environ['CUP_EXPERIMENT_GROUP_DIR'])/'isaac_geometry.json').read_text())
   assert physical['expected']['shape']==shape
   assert abs(float(physical['mass_kg'])-.08)<1e-6,'Cup mass changed'
  finally:self.destroy_client(c)
'''
assert anchor in s;s=s.replace(anchor,extra+anchor)
(R/'run_cases.py').write_text(s)
