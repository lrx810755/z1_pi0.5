import threading,unittest
from types import SimpleNamespace as N
from unittest.mock import patch
from wait_optimization import compact_route,preplan_lift,execute_lift

class Fake:
 def __init__(self):
  self._lock=threading.RLock();self._joint_positions={**{f'joint{i}':0. for i in range(1,7)},'jointGripper':-.5};self.gripper_open=-.9
  self.physical_test_lift_velocity_scale=.12;self.ee=(.4,0.,.1);self.ori=(0.,0.,0.,1.);self.object=(.5,0.,0.)
  self.events=[];self._object_poses={'apple':(100.,self.object)};self._object_orientations={'apple':(100.,self.ori)}
  self.geometry=N(check_path=lambda pts:self.events.append('geometry'))
 def _raise_if_stop_requested(self):pass
 def _current_ee_pose(self):return self.ee,self.ori
 def _current_object_position(self,obj):return self.object
 def get_logger(self):return N(info=lambda msg:self.events.append('log'))
 def _plan_candidate(self,pose,**kw):
  self.events.append('plan');return N(planned_trajectory=N(joint_trajectory=N(joint_names=['joint1'],points=[N(positions=[0.])])),trajectory_start=kw.get('start_state'))
 def _send_arm_trajectory(self,t):
  self.events.append('execute')
  if not getattr(self,'fail_to_lift',False):self.object=(.5,0.,.06)
  return False
 def _require_loaded_physical_grasp(self,*args,**kw):self.events.append('probe')

class Tests(unittest.TestCase):
 def setUp(self):
  self.n=Fake();p=patch('wait_optimization.time.monotonic',return_value=100.);p.start();self.addCleanup(p.stop)
 def cache(self):
  n=self.n;state=N(joint_state=N(name=[f'joint{i}' for i in range(1,7)]+['jointGripper'],position=[0.]*6+[-.9]))
  n.cached_lift=preplan_lift(n,{'position':n.ee,'orientation':n.ori},state,n.object,'attempt1');n.events.clear()
 def test_preplans_without_moving_then_reuses(self):
  self.cache();execute_lift(self.n,'attempt1')
  self.assertNotIn('plan',self.n.events);self.assertIn('execute',self.n.events)
  self.assertNotIn('probe',self.n.events) # The parent immediately performs the authoritative lift probe.
  self.assertIsNone(self.n.cached_lift)
 def test_changed_start_replans_before_execution(self):
  self.cache();self.n.ee=(.42,0.,.1);execute_lift(self.n,'attempt1')
  self.assertLess(self.n.events.index('plan'),self.n.events.index('execute'))
 def test_other_attempt_or_object_change_replans(self):
  self.cache();execute_lift(self.n,'attempt2');self.assertIn('plan',self.n.events)
  self.cache();self.n._object_poses['apple']=(100.,(.52,0.,0.));execute_lift(self.n,'attempt1');self.assertIn('plan',self.n.events)
 def test_low_lift_keeps_early_contact_guard(self):
  self.cache();self.n.fail_to_lift=True;execute_lift(self.n,'attempt1')
  self.assertIn('probe',self.n.events)
 def test_compacts_only_identical_consecutive_poses(self):
  step=lambda p,q:N(position=p,orientation=q)
  a=dict(position=[0,0,1],orientation=[0,0,0,1],stage='clear')
  b=dict(a,stage='rotate');c=dict(position=[0,0,0],orientation=[0,0,0,1],stage='descend')
  r=compact_route(dict(preparation=[a,b,c],recovery=[a,dict(a,orientation=[0,0,0,-1]),dict(a,position=[.0001,0,1])]))
  self.assertEqual([s['stage'] for s in r['preparation']],['clear','descend'])
  self.assertEqual(len(r['recovery']),2)
if __name__=='__main__':unittest.main()
