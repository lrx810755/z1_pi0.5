"""Exercise the actual place routine: cup releases directly; other objects descend."""
import ast,math,threading,unittest
from pathlib import Path
from types import SimpleNamespace
PATHS=[Path('/home/lp531/robot/cup_cylinder_horizontal200_20261108/base_assistant_snapshot.py'),Path('/home/lp531/robot/cup_horizontal_demo_20261108/base_assistant_snapshot.py'),Path('/home/lp531/ros2_ws/src/z1_ros2/z1_isaac_bridge/z1_isaac_bridge/grasp_assistant.py')]
def routine(path):
 tree=ast.parse(path.read_text());defs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in {'_guided_place','_validate_cup_direct_box_release'}]
 ns=dict(math=math,CUP_DIRECT_RELEASE_MIN_ROOT_Z_M=.115,GRASP_PROFILES={n:{'single_pass_box_alignment':True,'physical_test_lift_completion_tolerance_m':.005} for n in ['cup','beverage_bottle']},GRASP_ENVELOPE={},BOX_RELEASE_XY_TOLERANCE_M={'cup':.03,'beverage_bottle':.02},BOX_ALIGNMENT_MAX_STEPS=3,BOX_ALIGNMENT_STEP_M=.008,BOX_ALIGNMENT_MIN_PROGRESS_M=.001,BOX_ALIGNMENT_ORIENTATION_TOLERANCE_RAD=.025,BOX_FALLBACK_RELEASE_MAX_OBJECT_Z_M=.225)
 for node in defs:
  node.returns=None
  for arg in node.args.args:arg.annotation=None
 exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),str(path),'exec'),ns)
 return ns['_guided_place']
class Robot:
 def __init__(self,target='cup',z=.14,xy=(.32,.14),after_alignment_z=None,drift_before_release=None):
  self.target_object=target;self.position=(*xy,z);self.after_alignment_z=after_alignment_z;self.drift_before_release=drift_before_release;self._lock=threading.RLock();self._task_status={'physical_grasp_ok':True};self._verified_grasp_sequence=True;self._loaded_grasp_reference_local=None;self.box_approach_object_z=.14;self.box_release_object_z=.115;self.box_center=(.32,.14);self.calls=[];self._scene_grasp_status={}
 def get_logger(self):return SimpleNamespace(warning=lambda *a:None)
 def _require_loaded_physical_grasp(self,*args,**kwargs):
  self.calls.append(('verify',args[2]))
  if args[2] in ['到达蓝盒释放位后','杯子对中后直接释放前'] and self.drift_before_release:self.position=self.drift_before_release
 def _carry_loaded_object_toward_box(self,*a,**k):self.calls.append(('carry',))
 def _current_object_position(self,*a):return self.position
 def _move_object_to(self,p,label,**kw):
  self.calls.append(('move',label,p))
  self.position=tuple(p)
  if '水平对中' in label and self.after_alignment_z is not None:self.position=(*p[:2],self.after_alignment_z)
 def _set_status(self,*a,**k):pass
 def _set_grasp_control_state(self,**k):pass
 def _open_gripper_for_box_release(self,*a):self.calls.append(('open',))
 def _current_ee_pose(self):return ((.2,.1,.2),(0,0,0,1))
 def _execute_manual_pose(self,*a,**k):self.calls.append(('retreat',))
 def _execute_gripper(self,*a):pass
 def _command_scene_grasp(self,*a):pass
 def _wait_scene_grasp(self,*a):return True
 def _wait_task_flag(self,*a):return True
class Tests(unittest.TestCase):
 def test_cup_never_commands_descent_after_alignment(self):
  for p in PATHS:
   with self.subTest(source=str(p)):
    robot=Robot(xy=(.36,.14),after_alignment_z=.125);routine(p)(robot)
    self.assertIn(('open',),robot.calls)
    self.assertFalse(any(c[0]=='move' and c[1]=='垂直放入蓝盒' for c in robot.calls))
 def test_115mm_boundary_and_lower_heights_do_not_release(self):
  for z in [.114,.115]:
   robot=Robot(xy=(.36,.14),after_alignment_z=z)
   with self.assertRaises(RuntimeError):routine(PATHS[0])(robot)
   self.assertNotIn(('open',),robot.calls)
 def test_rechecks_pose_after_final_loaded_probe(self):
  for pose in [(.32,.14,.115),(.37,.14,.14),(.32,.14,float('nan'))]:
   robot=Robot(drift_before_release=pose)
   with self.assertRaises(RuntimeError):routine(PATHS[0])(robot)
   self.assertNotIn(('open',),robot.calls)
 def test_bottle_still_uses_existing_descent(self):
  robot=Robot(target='beverage_bottle');routine(PATHS[0])(robot)
  self.assertTrue(any(c[0]=='move' and c[1]=='垂直放入蓝盒' for c in robot.calls))
  self.assertIn(('open',),robot.calls)
if __name__=='__main__':unittest.main()
