"""Reuse production launch; isolate re-recording behavior in three node subclasses."""
import importlib.util
from pathlib import Path
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import SetLaunchConfiguration

HERE = Path(__file__).resolve().parent


def generate_launch_description():
    source = HERE / 'base.launch.py'
    spec = importlib.util.spec_from_file_location('rerecord_base_launch', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.get_package_share_path_original = module.get_package_share_path
    def factory(**kwargs):
        role = {'grasp_assistant': 'assistant', 'grasp_gui_bridge': 'gui', 'episode_recorder': 'recorder'}.get(kwargs.get('executable'))
        if role:
            kwargs.pop('package', None)
            kwargs['executable'] = str(HERE / 'nodes.py')
            kwargs['arguments'] = [role] + kwargs.get('arguments', [])
        return Node(**kwargs)
    module.Node = factory
    description = module.generate_launch_description()
    return LaunchDescription([
        SetLaunchConfiguration('isaac_script', str(HERE / 'scene.py')),
        *description.entities,
    ])
