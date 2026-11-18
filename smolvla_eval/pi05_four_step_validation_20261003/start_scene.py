import json,os
from pathlib import Path
p=Path(__file__).resolve().parent
env=os.environ.copy()
for key in ['PYTHONPATH','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH','CMAKE_PREFIX_PATH','ROS_VERSION']:
    env.pop(key,None)
env.update(json.loads((p/'scene_environment.json').read_text()))
args=['/home/lp531/isaac-sim/python.sh',
      '/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/scripts/run_z1_ros2_scene.py',
      '--scene','/home/lp531/ros2_ws/src/z1_ros2/isaac_assets/z1/generated_dual_camera/smolvla_dual_wrist_camera.usd',
      '--grasp-config','/home/lp531/ros2_ws/install/z1_isaac_bridge/share/z1_isaac_bridge/config/grasp_physics.json',
      '--eval-table-friction','0','--eval-arm-max-velocity','0']
os.execve(args[0],args,env)
