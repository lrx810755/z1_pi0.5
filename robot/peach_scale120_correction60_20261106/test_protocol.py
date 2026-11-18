import json
import tempfile
import unittest
from pathlib import Path
import core
from preflight import routes,trajectory_points
from fragment_core import partial_labels,valid_hold

class Protocol(unittest.TestCase):
 def test_plan_covers_peach_grid_without_bottle_height_offsets(self):
  tasks=core.make_tasks()
  self.assertEqual(len(tasks),60)
  from collections import Counter
  self.assertEqual(set(Counter(tuple(t['grid']) for t in tasks).values()),{3})
  self.assertTrue(all(0<=t['grid'][0]<4 and 0<=t['grid'][1]<5 for t in tasks))
  self.assertTrue(all(abs(t['offset_approach_m'][2])<=.004 for t in tasks))
  self.assertTrue(all(max(map(abs,t['rotation_tool_deg']))<=5 for t in tasks))
  self.assertEqual(len({t['id'] for t in tasks}),60)
 def test_nearly_vertical_peach_approach_has_stable_horizontal_axes(self):
  task=next(t for t in core.make_tasks() if t['kind']=='left')
  route=next(routes([.5,0,.2],[0,0,0,1],[.5,0,.019],task))
  self.assertGreater(route['bad_position'][1],0)
 def test_prepare_plans_high_approach_from_home_without_touching_reference(self):
  # A LIN from reset/home may be invalid at the joint3 upper stop. Reuse
  # the collision-checked OMPL approach before planning local corrections.
  import ast,copy,threading
  from types import SimpleNamespace as NS
  from scipy.spatial.transform import Rotation
  import numpy as np
  tree=ast.parse((Path(__file__).parent/'nodes.py').read_text())
  method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_generate_plan')
  class Base:
   def _generate_plan(self):return 'approved'
  # Supply the ROS message type for this isolated planner orchestration test.
  import sys,types
  from unittest.mock import patch
  moveit=types.ModuleType('moveit_msgs.msg')
  moveit.RobotState=lambda:NS(is_diff=True,joint_state=NS(name=[],position=[]))
  env=dict(copy=copy,time=__import__('time'),Rotation=Rotation,trajectory_points=trajectory_points,
           routes=routes,choose_route=__import__('preflight').choose_route,context=lambda:dict(task=core.make_tasks()[0]))
  klass=ast.ClassDef(name='Subject',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
  env['Base']=Base
  exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),'nodes.py','exec'),env)
  x=env['Subject']();x._lock=threading.Lock();x._joint_positions={f'joint{i}':0. for i in range(1,7)};x.gripper_open=-.9
  approved=NS(joint_trajectory=NS(joint_names=[f'joint{i}' for i in range(1,7)],points=[NS(positions=[0,-.5,-1.,0,0,0])]))
  x._approved_trajectory=approved
  x.geometry=NS(link_fk=lambda q:(np.array([.5,0,.15]),np.eye(3)),check_path=lambda pts:None)
  x._current_object_position=lambda _: [.5,0,0];x._raise_if_stop_requested=lambda:None
  x.physical_test_lift_velocity_scale=.12
  x._object_poses={'peach':(0.,(.5,0,0))};x._object_orientations={'peach':(0.,(0,0,0,1))}
  starts=[]
  def plan(candidate,**kwargs):
   starts.append(list(kwargs['start_state'].joint_state.position))
   return NS(planned_trajectory=approved)
  x._plan_candidate=plan
  x._state_after_trajectory=lambda result: NS(is_diff=True,joint_state=NS(name=[f'joint{i}' for i in range(1,7)]+['jointGripper'],position=[0,-.5,-1.,0,0,0,-.9]))
  with patch.dict(sys.modules,{'moveit_msgs.msg':moveit}):x._generate_plan()
  self.assertEqual(starts[0][2],0.)
  self.assertIsNotNone(x.cached_lift)
  self.assertEqual(x.cached_lift['position'],(.5,0.,.15))
  route,planned,_=x.cached
  self.assertEqual(route['preparation'][0]['stage'],'prepare_clearance')
  self.assertFalse(route['preparation'][0]['linear'])
  self.assertEqual(planned[0][1].joint_trajectory.points[-1].positions,approved.joint_trajectory.points[-1].positions)
 def test_split_is_by_cell(self):
  cells={}
  for t in core.make_tasks():cells.setdefault(tuple(t['grid']),set()).add(t['split'])
  self.assertTrue(all(len(v)==1 for v in cells.values()))
 def test_low_keeps_requested_error_and_recovers_height_before_rotation(self):
  task=next(t for t in core.make_tasks() if t['kind']=='low')
  r=next(routes([.3,0,.1],[0,0,0,1],[.5,0,.019],task))
  self.assertAlmostEqual(r['bad_position'][2]-.1,task['offset_approach_m'][2])
  self.assertEqual(r['recovery'][0]['stage'],'lift_clear')
  self.assertEqual(r['recovery'][1]['stage'],'correct_orientation')
  self.assertLessEqual(r['lift_m'],.025)
 def test_labels_use_close_command_not_late_phase(self):
  rows=[dict(phase='READY',action=[0]*6+[-.9]) for _ in range(3)]
  rows += [dict(phase='READY',action=[0]*6+[-.85]) for _ in range(20)]
  labels,onset,last=partial_labels(rows)
  self.assertEqual(onset,3);self.assertEqual(labels,[0]*3+[1]*20)
  self.assertEqual(last,7)
 def test_reopening_rejects_fragment(self):
  rows=[dict(action=[0]*6+[x]) for x in [-.9,-.85,-.8,-.9]]
  with self.assertRaises(ValueError):partial_labels(rows)
 def test_hold_requires_real_contact_and_object_clear_of_table(self):
  good=dict(target_object='peach',object_position_m=[.5,0,.06],physical_grasp_ok=True,lifted=True,bilateral_contact=True,
   target_collisions_enabled=True,frozen=False,object_touching_table=False,left_finger_touching_table=False,right_finger_touching_table=False)
  self.assertTrue(valid_hold(good))
  for key,val in [('bilateral_contact',False),('object_touching_table',True),('target_object','beverage_bottle')]:
   self.assertFalse(valid_hold(dict(good,**{key:val})))
 def test_completed_requires_peach_evidence_and_frames(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d)/'episode_000000';r.mkdir()
   (r/'correction.json').write_text(json.dumps(dict(accepted=True,plan_hash='x',task_id='a')))
   (r/'episode_summary.json').write_text(json.dumps(dict(fragment_success=True,physical_grasp_ok=True,target_object='beverage_bottle')))
   (r/'frames.jsonl').write_text('{}\n')
   self.assertEqual(core.completed_ids(d,'x'),set())
if __name__=='__main__':unittest.main()
