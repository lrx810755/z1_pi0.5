"""Private bottle-only forward loaded transport with preflight and live guards."""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from forward_core import ForwardGuard,GuardViolation,grasp_offset,ee_target
BOTTLE_TRANSPORT_Z=float(json.loads(Path(__file__).with_name('grasp_physics.json').read_text())['object_grasp_profiles']['beverage_bottle']['transport_object_z_m'])
LOG=Path('/home/lp531/z1_queue_logs/bottle_horizontal_supplement_20261109')

class ForwardTransport:
 def __init__(self,*args,**kwargs):
  self._loaded_guard=None;self._forward_joint_at=0.;self._forward_contact_at=0.
  super().__init__(*args,**kwargs)
 def _joint_callback(self,message):
  super()._joint_callback(message);self._forward_joint_at=time.monotonic()
 def _scene_grasp_status_callback(self,message):
  super()._scene_grasp_status_callback(message)
  try:
   if json.loads(message.data).get('phase')=='CONTACT_STATE':self._forward_contact_at=time.monotonic()
  except (ValueError,TypeError):pass
 def _forward_pose(self):
  with self._lock:q=[self._joint_positions[f'joint{i}'] for i in range(1,7)]
  return self.geometry.link_fk(q)
 def _audit_forward_trajectory(self,trajectory,start,goal,rotation,offset,phase,expected_start_q=None):
  jt=trajectory.joint_trajectory;order=[jt.joint_names.index(f'joint{i}') for i in range(1,7)]
  q=[np.asarray([p.positions[i] for i in order]) for p in jt.points]
  if not q:raise GuardViolation('empty loaded trajectory')
  guard=ForwardGuard(start,goal,rotation,phase);samples=0
  with self._lock:
   actual=np.array([self._joint_positions[f'joint{i}'] for i in range(1,7)]);gripper=self._joint_positions['jointGripper']
  expected=actual if expected_start_q is None else np.asarray(expected_start_q)
  if np.max(np.abs(q[0]-expected))>.01:raise GuardViolation('loaded trajectory has stale start')
  wrist_guard=None
  if phase=='carry':
   wrist_start,wrist_rotation=self.geometry.link_fk(q[0]);wrist_goal,_=self.geometry.link_fk(q[-1]);wrist_guard=ForwardGuard(wrist_start,wrist_goal,wrist_rotation,'carry',carry_max_height=.500)
  previous=q[0]
  for end in q:
   for alpha in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(end-previous))/.005))+1)):
    joint=previous+alpha*(end-previous);p,r=self.geometry.link_fk(joint);guard.check(p+r@offset,r)
    if wrist_guard is not None:wrist_guard.check(p,r)
    if self.geometry.height(list(joint)+[gripper])<.003:raise GuardViolation('loaded fingers below3mm table floor')
    samples+=1
   previous=end
  p,r=self.geometry.link_fk(q[-1]);error=float(np.linalg.norm(p+r@offset-np.asarray(goal)))
  if error>.003:raise GuardViolation(f'object endpoint error {error:.5f}m')
  return dict(samples=samples,object_endpoint_error_m=error,phase=phase,reverse_tolerance_m=.005,lift_lateral_tolerance_m=.005,lateral_tolerance_m=.020,orientation_tolerance_rad=(.45 if phase=='carry' else .035))
 def _forward_segment(self,goal,phase):
  target='beverage_bottle';start=np.asarray(self._current_object_position(target),float)
  ee,quaternion=self._current_ee_pose();rotation=Rotation.from_quat(quaternion).as_matrix();offset=grasp_offset(ee,rotation,start)
  target_ee=ee_target(goal,rotation,offset);errors=[]
  # No unguarded fallback: both LIN and OMPL are audited against the same corridor.
  for linear,velocity,acceleration in [(True,.04,.02),(True,.02,.01),(False,.04,.02)]:
   self._raise_if_stop_requested()
   self._set_status('JOG_PLANNING',f'瓶子前向搬运：{phase}，'+('LIN' if linear else 'OMPL受限候选'),False)
   try:
    result=self._plan_candidate(dict(position=target_ee.tolist(),orientation=list(quaternion)),linear=linear,position_tolerance=.001,orientation_tolerance=.01,velocity_scale=velocity,acceleration_scale=acceleration)
    audit=self._audit_forward_trajectory(result.planned_trajectory,start,goal,rotation,offset,phase)
    if np.linalg.norm(np.asarray(self._current_object_position(target))-start)>.003:raise GuardViolation('bottle moved during planning; refusing stale loaded path')
   except (RuntimeError,ValueError) as exc:
    errors.append(str(exc));self.get_logger().warning('前向带载候选被拒绝：'+str(exc));continue
   LOG.mkdir(parents=True,exist_ok=True)
   (LOG/f'forward_plan_{time.time_ns()}.json').write_text(json.dumps(dict(object_start_m=start.tolist(),object_goal_m=list(goal),ee_goal_m=target_ee.tolist(),orientation_xyzw=list(quaternion),grasp_offset_in_ee_m=offset.tolist(),linear=linear,velocity_scale=velocity,acceleration_scale=acceleration,audit=audit,rejected=errors),ensure_ascii=False,indent=2))
   self._loaded_guard=dict(guard=ForwardGuard(start,goal,rotation,phase),offset=offset,last_contact_ok=time.monotonic())
   try:
    self._set_status('JOGGING',f'瓶子前向搬运：{phase}',False)
    if self._send_arm_trajectory(result.planned_trajectory):raise RuntimeError('Forward loaded transport stopped')
   finally:self._loaded_guard=None
   return
  raise RuntimeError('没有满足前向、姿态及高度约束的带载轨迹：'+' | '.join(errors))
 def _move_object_to(self,destination,description,linear,*args,**kwargs):
  if self.target_object=='beverage_bottle' and '水平对中' in description:
   return self._forward_segment(destination,'horizontal')
  return super()._move_object_to(destination,description,linear,*args,**kwargs)
 def _carry_loaded_object_toward_box(self,target_object,loaded_envelope,minimum_object_z,stage,require_box_approach=False,preferred_object_z=None):
  if target_object!='beverage_bottle':return super()._carry_loaded_object_toward_box(target_object,loaded_envelope,minimum_object_z,stage,require_box_approach,preferred_object_z)
  def verify(label):return self._require_loaded_physical_grasp(target_object,loaded_envelope,label,use_verified_lift_reference=False,loaded_envelope_margin_m=self.transport_loaded_grasp_guard_margin)
  position=np.asarray(self._current_object_position(target_object),float);box=np.asarray(self.box_center,float)
  if position[2]>=minimum_object_z and (not require_box_approach or np.linalg.norm(position[:2]-box)<=.045):return verify(stage)
  verify('前向带载开始前')
  height=max(float(minimum_object_z),float(self.box_approach_object_z),BOTTLE_TRANSPORT_Z,float(preferred_object_z or minimum_object_z))
  if position[2]<height-.002:
   self._forward_segment([float(position[0]),float(position[1]),height],'lift');verify('前向竖直抬升后')
  position=np.asarray(self._current_object_position(target_object),float)
  if position[2]<minimum_object_z:raise GuardViolation('forward lift did not reach safe object height')
  if require_box_approach and np.linalg.norm(position[:2]-box)>.045:
   self._forward_segment([float(box[0]),float(box[1]),float(position[2])],'horizontal')
  probe=verify(stage);position=np.asarray(self._current_object_position(target_object),float)
  if position[2]<minimum_object_z or (require_box_approach and np.linalg.norm(position[:2]-box)>.045):raise GuardViolation('forward carry did not reach box acceptance corridor')
  return probe
 def _forward_live_check(self):
  context=self._loaded_guard;now=time.monotonic()
  if now-self._forward_joint_at>1.:raise GuardViolation('loaded joint feedback stale')
  if now-self._forward_contact_at>1.:raise GuardViolation('loaded contact feedback stale')
  obj=np.asarray(self._current_object_position('beverage_bottle'),float);p,r=self._forward_pose();context['guard'].check(obj,r)
  if context.get('wrist_guard') is not None:context['wrist_guard'].check(p,r)
  if np.linalg.norm(grasp_offset(p,r,obj)-context['offset'])>self.physical_test_lift_relative_position_tolerance:raise GuardViolation('loaded bottle slip exceeded configured relative-pose limit')
  with self._lock:contact=dict(self._scene_grasp_contact_status)
  good=bool(contact.get('contact_report_available') and contact.get('left_finger_contact_object')=='beverage_bottle' and contact.get('right_finger_contact_object')=='beverage_bottle')
  if good:context['last_contact_ok']=now
  elif now-context['last_contact_ok']>.5:raise GuardViolation('loaded bottle lost bilateral contact for0.5s')
 def _send_action_goal(self,client,goal,timeout_s):
  if getattr(self,'_loaded_guard',None) is None or client is not self._trajectory_client:return super()._send_action_goal(client,goal,timeout_s)
  self._raise_if_stop_requested();handle=self._wait_future(client.send_goal_async(goal),5.)
  if handle is None or not handle.accepted:raise RuntimeError('Forward transport arm action rejected')
  with self._lock:self._active_goal_handle=handle
  future=handle.get_result_async();deadline=time.monotonic()+timeout_s
  try:
   while not future.done():
    self._raise_if_stop_requested();self._forward_live_check()
    if time.monotonic()>deadline:raise GuardViolation('forward loaded action timeout')
    time.sleep(.02)
   return future.result().result
  except BaseException:
   self._wait_future(handle.cancel_goal_async(),2.)
   self._wait_future(future,2.)
   raise
  finally:
   with self._lock:self._active_goal_handle=None
