"""Keep each native expert launch; replace only task setup and recorder nodes."""
import importlib.util
import os
from pathlib import Path
from launch_ros.actions import Node


def generate_launch_description():
    source=Path(os.environ['Z1_REVERSE8_NATIVE'])/'collection.launch.py'
    spec=importlib.util.spec_from_file_location('reverse8_native_launch',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def factory(**kwargs):
        executable=kwargs.get('executable')
        args=kwargs.get('arguments',[])
        role='gui' if (executable=='grasp_gui_bridge' or any(str(a).endswith('gui_complete_search.py') for a in args)) else 'recorder' if (
            executable=='episode_recorder' or any(str(a).endswith('failure_recorder.py') for a in args)) else None
        if role:
            kwargs.pop('package',None);kwargs['executable']='/usr/bin/python3'
            kwargs['arguments']=[str(Path(__file__).with_name('basic_nodes.py')),role]
        return Node(**kwargs)
    module.Node=factory
    return module.generate_launch_description()
