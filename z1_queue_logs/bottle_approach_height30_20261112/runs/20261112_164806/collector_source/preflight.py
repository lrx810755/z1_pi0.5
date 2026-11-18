"""Construct the real ROS assistant and inspect launch wiring without a scene."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import nodes,horizontal50
horizontal50.base.main=lambda:None
nodes.assistant_main()
import rclpy
rclpy.init(args=['--ros-args','-p','target_object:=beverage_bottle'])
node=None
try:
 node=horizontal50.base.GraspAssistant()
 assert node.target_object=='beverage_bottle'
 assert node.geometry.height([0,0,0,0,0,0,-.9])>0
finally:
 if node is not None:node.destroy_node()
 rclpy.shutdown()
spec=importlib.util.spec_from_file_location('coverage30launch',ROOT/'collection.launch.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);ld=module.generate_launch_description()
from launch_ros.actions import Node
roles=[]
for n in ld.entities:
 if isinstance(n,Node):
  text=str(vars(n).get('_Node__node_executable',[]))
  if 'nodes.py' in text:roles.append(text)
assert len(roles)==3,roles
print(json.dumps(dict(passed=True,assistant_constructed=True,isolated_nodes=3,scene_started=False)))
