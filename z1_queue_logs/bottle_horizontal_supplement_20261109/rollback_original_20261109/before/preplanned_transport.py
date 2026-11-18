"""Bottle look-ahead: validate both closed-gripper legs before any lift."""
import json,time
import numpy as np
from scipy.spatial.transform import Rotation
from moveit_msgs.msg import RobotState
from forward_core import ForwardGuard,GuardViolation,grasp_offset,ee_target
from forward_transport import ForwardTransport,BOTTLE_TRANSPORT_Z,LOG

class PreplannedBottleTransport(ForwardTransport):
 def _closed_transport_state(self,q,gripper):
  state=RobotState();state.is_diff=True
  state.joint_state.name=[f'joint{i}' for i in range(1,7)]+['jointGripper']
  state.joint_state.position=[float(v) for v in q]+[float(gripper)]
  state.joint_state.velocity=[0.]*7
  return state

 def _plan_loaded_leg(self,start,goal,rotation,offset,phase,q,gripper):
  quaternion=Rotation.from_matrix(rotation).as_quat().tolist();target=ee_target(goal,rotation,offset);errors=[]
  state=self._closed_transport_state(q,gripper)
  for linear,velocity,acceleration in [(True,.04,.02),(True,.02,.01),(False,.04,.02)]:
   self._raise_if_stop_requested()
   self._set_status('JOG_PLANNING',f'闭爪后提前规划：{phase}，'+('LIN' if linear else 'OMPL受限候选'),False)
   try:
    result=self._plan_candidate(dict(position=target.tolist(),orientation=quaternion),linear=linear,position_tolerance=.001,orientation_tolerance=.01,velocity_scale=velocity,acceleration_scale=acceleration,start_state=state)
    audit=self._audit_forward_trajectory(result.planned_trajectory,start,goal,rotation,offset,phase,expected_start_q=q)
   except (RuntimeError,ValueError) as exc:
    errors.append(str(exc));self.get_logger().warning('闭爪后预规划候选被拒绝：'+str(exc));continue
   jt=result.planned_trajectory.joint_trajectory
   end=np.asarray([jt.points[-1].positions[jt.joint_names.index(f'joint{i}')] for i in range(1,7)])
   return dict(phase=phase,start=np.asarray(start),goal=np.asarray(goal),rotation=rotation,offset=offset,start_q=np.asarray(q),end_q=end,trajectory=result.planned_trajectory,audit=audit,linear=linear,rejected=errors)
  raise RuntimeError(f'闭爪后{phase}提前规划失败（尚未抬升）：'+' | '.join(errors))

 def _plan_loaded_pair(self,position,height,require_box):
  with self._lock:
   q=np.array([self._joint_positions[f'joint{i}'] for i in range(1,7)]);gripper=self._joint_positions['jointGripper']
  ee,rotation=self.geometry.link_fk(q);offset=grasp_offset(ee,rotation,position)
  start=np.asarray(position);route=[];started=time.monotonic()
  if start[2]<height-.002:
   goal=np.array([start[0],start[1],height])
   leg=self._plan_loaded_leg(start,goal,rotation,offset,'lift',q,gripper);route.append(leg)
   q=leg['end_q'];p,r=self.geometry.link_fk(q);start=p+r@offset
  if require_box and np.linalg.norm(start[:2]-np.asarray(self.box_center))>.020:
   goal=np.array([*self.box_center,start[2]])
   leg=self._plan_loaded_leg(start,goal,rotation,offset,'horizontal',q,gripper);route.append(leg)
  # Both plans exist before any arm command. Keep one rigid grasp reference.
  if np.linalg.norm(np.asarray(self._current_object_position('beverage_bottle'))-position)>.003:
   raise GuardViolation('stale bottle pose after pair preplanning; no lift executed')
  with self._lock:actual=np.array([self._joint_positions[f'joint{i}'] for i in range(1,7)])
  if route and np.max(np.abs(actual-route[0]['start_q']))>.01:
   raise GuardViolation('stale joints after pair preplanning; no lift executed')
  LOG.mkdir(parents=True,exist_ok=True)
  audit_path=LOG/f'preplanned_pair_{time.time_ns()}.json'
  audit_path.write_text(json.dumps(dict(all_planned_before_first_motion=True,planning_duration_s=time.monotonic()-started,gripper_position_rad=gripper,legs=[dict(phase=l['phase'],object_start_m=l['start'].tolist(),object_goal_m=l['goal'].tolist(),start_q=l['start_q'].tolist(),end_q=l['end_q'].tolist(),audit=l['audit'],linear=l['linear'],rejected=l['rejected']) for l in route]),ensure_ascii=False,indent=2))
  self.get_logger().info(f'闭爪后两段预规划完成，共{len(route)}段；后续不在抬升与搬运之间重新规划：{audit_path}')
  return route

 def _execute_loaded_leg(self,leg):
  self._raise_if_stop_requested()
  with self._lock:q=np.array([self._joint_positions[f'joint{i}'] for i in range(1,7)])
  if np.max(np.abs(q-leg['start_q']))>.01:raise GuardViolation('stale preplanned loaded joint start; no replanning')
  position=np.asarray(self._current_object_position('beverage_bottle'))
  if np.linalg.norm(position-leg['start'])>.003:raise GuardViolation('stale preplanned bottle start; no replanning')
  self._loaded_guard=dict(guard=ForwardGuard(leg['start'],leg['goal'],leg['rotation'],leg['phase']),offset=leg['offset'],last_contact_ok=time.monotonic())
  if leg['phase']=='carry':
   wrist_start,wrist_rotation=self.geometry.link_fk(leg['start_q']);wrist_goal,_=self.geometry.link_fk(leg['end_q'])
   self._loaded_guard['wrist_guard']=ForwardGuard(wrist_start,wrist_goal,wrist_rotation,'carry',carry_max_height=.500)
  try:
   self._set_status('JOGGING',f'执行瓶子已预规划轨迹：{leg["phase"]}',False)
   if self._send_arm_trajectory(leg['trajectory']):raise RuntimeError('preplanned loaded transport stopped')
  finally:self._loaded_guard=None

 def _carry_loaded_object_toward_box(self,target_object,loaded_envelope,minimum_object_z,stage,require_box_approach=False,preferred_object_z=None):
  if target_object!='beverage_bottle':return super()._carry_loaded_object_toward_box(target_object,loaded_envelope,minimum_object_z,stage,require_box_approach,preferred_object_z)
  def verify(label):return self._require_loaded_physical_grasp(target_object,loaded_envelope,label,use_verified_lift_reference=False,loaded_envelope_margin_m=self.transport_loaded_grasp_guard_margin)
  position=np.asarray(self._current_object_position(target_object),float);box=np.asarray(self.box_center)
  # Reach final XY acceptance up front so placement does not add a new alignment plan.
  if position[2]>=minimum_object_z and (not require_box_approach or np.linalg.norm(position[:2]-box)<=.020):return verify(stage)
  verify('闭爪后整段预规划前')
  height=max(float(minimum_object_z),float(self.box_approach_object_z),BOTTLE_TRANSPORT_Z,float(preferred_object_z or minimum_object_z))
  route=self._plan_loaded_pair(np.asarray(self._current_object_position(target_object),float),height,require_box_approach)
  verify('整段预规划后、执行前')
  for leg in route:
   self._execute_loaded_leg(leg)
   verify('已预规划'+leg['phase']+'执行后')
  position=np.asarray(self._current_object_position(target_object))
  if position[2]<minimum_object_z or (require_box_approach and np.linalg.norm(position[:2]-box)>.020):raise GuardViolation('preplanned carry did not reach box acceptance corridor')
  return verify(stage)
