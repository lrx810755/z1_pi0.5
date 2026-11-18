"""Reuse the production launch and replace only collection-specific nodes."""
import importlib.util
from pathlib import Path
from launch_ros.actions import Node

HERE=Path(__file__).resolve().parent

def generate_launch_description():
    source=HERE.parents[1]/'z1_isaac_bridge/launch/z1_isaac.launch.py'
    spec=importlib.util.spec_from_file_location('base_collection_launch',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def factory(**kwargs):
        role={'grasp_assistant':'assistant','grasp_gui_bridge':'gui','episode_recorder':'recorder'}.get(kwargs.get('executable'))
        if role:
            kwargs.pop('package',None);kwargs['executable']=str(HERE/'nodes.py')
            kwargs['arguments']=[role]+kwargs.get('arguments',[])
        return Node(**kwargs)
    module.Node=factory
    return module.generate_launch_description()
