import copy,threading,unittest
from types import SimpleNamespace as N
from unittest.mock import patch
from wait_optimization import AppleWaitOptimization

class Backend:
 def __init__(self):
  self._lock=threading.RLock();self.target_object='apple';self._stop_requested=False
  self._joint_positions={'joint1':0.,'jointGripper':-.5}
  self._object_poses={'apple':(100.,(.5,0.,0.)), 'cup':(100.,(.6,.1,0.))}
  self._object_orientations={k:(100.,(0.,0.,0.,1.)) for k in self._object_poses}
  self._grasp_candidate={'position':(.4,0.,.1),'orientation':(0.,0.,0.,1.)}
  self.ee=(.4,0.,.1);self.ori=(0.,0.,0.,1.);self.events=[];self.clock=100.
  self._task_status={'bilateral_contact':True,'physical_grasp_ok':True,'left_finger_touching_table':False,'right_finger_touching_table':False}
  self._task_status_seen_at=100.
 def _current_ee_pose(self):return self.ee,self.ori
 def _current_object_position(self,obj):return self._object_poses[obj][1]
 def _raise_if_stop_requested(self):
  if self._stop_requested:raise RuntimeError('stopped')
 def get_logger(self):return N(info=lambda s:self.events.append(('log',s)))
 def _generate_plan(self):
  self._plan_apple_transfer((.5,0.,0.),(.3,0.,.1),self.ori,.14,start_state='wrong')
  self.selected=self._plan_apple_transfer((.5,0.,0.),self.ee,self.ori,.14,start_state='selected')
  return 'ready'
 def _plan_apple_transfer(self,origin,ee,ori,height,start_state=None):
  self.events.append(('plan',start_state))
  return [dict(phase='lift',height=height,trajectory=N(joint_trajectory=N(joint_names=['joint1'],points=[N(positions=[0.])])))]
 def _require_loaded_physical_grasp(self,*args,**kwargs):
  self.events.append(('probe',args[2]));return {'verified':True}
 def _task_status_callback(self,message):self._task_status=message
 def _reset_assistant_service(self,*args):self._grasp_candidate=None
 def _target_callback(self,message):self.target_object=message
 def _force_stop_service(self,*args):self._stop_requested=True

class Node(AppleWaitOptimization,Backend):pass
class Tests(unittest.TestCase):
 def setUp(self):
  self.n=Node();self.timepatch=patch('wait_optimization.time.monotonic',side_effect=lambda:self.n.clock);self.timepatch.start();self.addCleanup(self.timepatch.stop)
 def prepare(self):self.n._generate_plan();self.n.events.clear()
 def test_selected_preplan_is_reused_once(self):
  self.prepare();n=self.n
  self.assertEqual(n._plan_apple_transfer((.5,0.,0.),n.ee,n.ori,.14),n.selected)
  self.assertFalse(any(x[0]=='plan' for x in n.events))
  n._plan_apple_transfer((.5,0.,0.),n.ee,n.ori,.14)
  self.assertEqual(sum(x[0]=='plan' for x in n.events),1)
 def test_changed_start_or_scene_or_goal_replans(self):
  mutations=[lambda n:setattr(n,'ee',(.42,0.,.1)),lambda n:n._joint_positions.update(joint1=.04),lambda n:n._object_poses.update(cup=(100.,(.62,.1,0.))),lambda n:setattr(n,'clock',131.),lambda n:setattr(n,'_grasp_candidate',copy.deepcopy(n._grasp_candidate))]
  for change in mutations:
   self.n=Node();self.prepare();change(self.n);n=self.n
   n._plan_apple_transfer((.5,0.,0.),n.ee,n.ori,.14)
   self.assertTrue(any(x[0]=='plan' for x in n.events))
 def test_changed_height_replans(self):
  self.prepare();n=self.n;n._plan_apple_transfer((.5,0.,0.),n.ee,n.ori,.16)
  self.assertTrue(any(x[0]=='plan' for x in n.events))
 def test_adjacent_probe_reused_but_postcheck_always_runs(self):
  n=self.n;n._require_loaded_physical_grasp('apple',{},'苹果已规划路线执行后',use_verified_lift_reference=False)
  n._require_loaded_physical_grasp('apple',{},'苹果已规划路线执行前',use_verified_lift_reference=False)
  n._require_loaded_physical_grasp('apple',{},'苹果已规划路线执行后',use_verified_lift_reference=False)
  self.assertEqual([x for x in n.events if x[0]=='probe'],[('probe','苹果已规划路线执行后')]*2)
 def test_aged_or_lost_contact_or_changed_joint_requires_new_probe(self):
  changes=[lambda n:setattr(n,'clock',100.4),lambda n:n._task_status.update(bilateral_contact=False),lambda n:n._joint_positions.update(joint1=.01),lambda n:n._object_poses.update(apple=(100.,(.51,0.,0.)))]
  for change in changes:
   self.n=Node();n=self.n
   n._require_loaded_physical_grasp('apple',{},'苹果已规划路线执行后',use_verified_lift_reference=False)
   change(n);n._require_loaded_physical_grasp('apple',{},'苹果已规划路线执行前',use_verified_lift_reference=False)
   self.assertEqual(sum(x[0]=='probe' for x in n.events),2)
 def test_reset_invalidates_both_caches(self):
  self.prepare();n=self.n;n._require_loaded_physical_grasp('apple',{},'苹果已规划路线执行后',use_verified_lift_reference=False)
  n._reset_assistant_service(None,None)
  self.assertIsNone(n._apple_preplanned_transfer);self.assertIsNone(n._apple_adjacent_probe)
 def test_stop_is_never_bypassed(self):
  self.prepare();n=self.n;n._stop_requested=True
  with self.assertRaises(RuntimeError):n._plan_apple_transfer((.5,0.,0.),n.ee,n.ori,.14)
if __name__=='__main__':unittest.main()
