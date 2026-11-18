"""Import and construct the actual native ROS assistant without a simulator."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
stage=sys.argv[1];plan=json.loads((ROOT/'plan.json').read_text());entry=next(s for s in plan['stages'] if s['id']==stage)
native=Path(entry['native']);sys.path.insert(0,str(native));os.environ['Z1_ISAAC_GRASP_CONFIG']=str(native/'grasp_physics.json')
if stage=='bottle_basic':
 import horizontal50 as module
 cls=module.HorizontalAssistant
else:
 import assistant as module
 cls=module.AppleAssistant if stage=='apple_basic' else module.HorizontalAssistant
import rclpy
rclpy.init(args=['--ros-args','-p','target_object:='+entry['object']])
node=None
try:
 node=cls();assert node.target_object==entry['object']
 if stage=='apple_basic':assert module.base.GRASP_PROFILES['apple']['candidate_count']==16
 print(json.dumps(dict(passed=True,stage=stage,target=node.target_object,assistant_constructed=True,scene_started=False)))
finally:
 if node is not None:node.destroy_node()
 rclpy.shutdown()
