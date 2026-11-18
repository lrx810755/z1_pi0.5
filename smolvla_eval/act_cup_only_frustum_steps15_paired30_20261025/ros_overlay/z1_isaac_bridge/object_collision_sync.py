"""Experiment-only collision synchronization for the sole remaining object."""
import importlib.util
import sys
from pathlib import Path
source = Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/object_collision_sync.py')
spec = importlib.util.spec_from_file_location('z1_isaac_bridge._full_object_collision_sync', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.OBJECT_COLLISIONS = {'cup': module.OBJECT_COLLISIONS['cup']}
ObjectCollisionSync = module.ObjectCollisionSync
OBJECT_COLLISIONS = module.OBJECT_COLLISIONS
main = module.main
