import importlib.util
from pathlib import Path
from launch_ros.actions import Node
from launch.conditions import IfCondition
HERE=Path(__file__).parent

def generate_launch_description():
 source=Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/launch/z1_isaac.launch.py')
 spec=importlib.util.spec_from_file_location('base_probe_launch',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 original_share=m.get_package_share_path
 m.get_package_share_path=lambda name:HERE/'share' if name=='z1_isaac_bridge' else original_share(name)
 def factory(**kwargs):
  kwargs['additional_env']=dict(kwargs.get('additional_env',{}),Z1_ISAAC_GRASP_CONFIG=str(HERE/'grasp_physics.json'))
  if kwargs.get('executable')=='moveit_servo_vendor':kwargs['condition']=IfCondition('false')
  if kwargs.get('executable')=='grasp_assistant':
   kwargs.pop('package',None);kwargs['executable']=str(HERE/'assistant.py')
  return Node(**kwargs)
 m.Node=factory
 return m.generate_launch_description()
