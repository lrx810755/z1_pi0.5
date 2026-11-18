"""Cup-style diagonal loaded carry with monotone box-distance and height guards."""
import json,time
import numpy as np
from scipy.spatial.transform import Rotation
from preplanned_transport import PreplannedBottleTransport
from forward_transport import BOTTLE_TRANSPORT_Z,LOG
from forward_core import GuardViolation,grasp_offset,ee_target

class CombinedBottleTransport(PreplannedBottleTransport):
 def _plan_loaded_pair(self,position,height,require_box):
  with self._lock:
   initial_q=np.array([self._joint_positions[f'joint{i}'] for i in range(1,7)]);gripper=self._joint_positions['jointGripper']
  ee,initial_rotation=self.geometry.link_fk(initial_q);offset=grasp_offset(ee,initial_rotation,position)
  goal=np.array([*(self.box_center if require_box else position[:2]),max(height,position[2])]);errors=[];started=time.monotonic()
  # A link-frame target-point offset constrains the bottle root while the wrist rotates.
  for tolerance in (.2,.4):
   self._raise_if_stop_requested();rotation=initial_rotation
   self._set_status('JOG_PLANNING','闭爪后提前规划向蓝盒斜向抬升搬运',False)
   try:
    result=self._plan_candidate(dict(position=goal.tolist(),orientation=Rotation.from_matrix(rotation).as_quat().tolist(),target_point_offset=offset.tolist(),position_z_range=[float(goal[2]),.300]),linear=False,position_tolerance=.001,orientation_tolerance=tolerance,velocity_scale=.08,acceleration_scale=.04,start_state=self._closed_transport_state(initial_q,gripper))
    jt=result.planned_trajectory.joint_trajectory;end=np.array([jt.points[-1].positions[jt.joint_names.index(f'joint{i}')] for i in range(1,7)])
    endpoint,endpoint_rotation=self.geometry.link_fk(end);object_endpoint=endpoint+endpoint_rotation@offset
    if not goal[2]-.001<=object_endpoint[2]<=.300:raise GuardViolation('planned endpoint outside safe object height range')
    accepted_goal=np.array([goal[0],goal[1],object_endpoint[2]])
    audit=self._audit_forward_trajectory(result.planned_trajectory,position,accepted_goal,initial_rotation,offset,'carry',expected_start_q=initial_q)

   except (RuntimeError,ValueError) as exc:
    errors.append(str(exc));self.get_logger().warning('斜向前上搬运候选被拒绝：'+str(exc));continue
   if np.linalg.norm(np.asarray(self._current_object_position('beverage_bottle'))-position)>.003:raise GuardViolation('stale bottle pose after combined planning')
   leg=dict(phase='carry',start=np.asarray(position),goal=accepted_goal,rotation=initial_rotation,offset=offset,start_q=initial_q,end_q=end,trajectory=result.planned_trajectory,audit=audit,linear=False,rejected=errors)
   LOG.mkdir(parents=True,exist_ok=True)
   path=LOG/f'combined_carry_{time.time_ns()}.json';path.write_text(json.dumps(dict(object_start_m=list(position),object_goal_m=accepted_goal.tolist(),object_goal_height_range_m=[float(goal[2]),.300],planning_duration_s=time.monotonic()-started,goal_orientation_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),audit=audit,requested_orientation_tolerance_rad=tolerance,rejected=errors),ensure_ascii=False,indent=2))
   self.get_logger().info('闭爪后向蓝盒斜向抬升搬运整条轨迹预规划完成：'+str(path))
   return [leg]
  raise RuntimeError('向蓝盒前上方搬运无可用轨迹（尚未执行）：'+' | '.join(errors))

 def _move_object_to(self,destination,description,linear,*args,**kwargs):
  if self.target_object=='beverage_bottle' and description=='垂直放入蓝盒':
   position=self._current_object_position('beverage_bottle')
   if .115<=position[2]<=.130:return
   destination=(destination[0],destination[1],.1225)
   try:return super()._move_object_to(destination,description,linear,*args,**kwargs)
   except RuntimeError:
    if .115<=self._current_object_position('beverage_bottle')[2]<=.130:return
    raise
  return super()._move_object_to(destination,description,linear,*args,**kwargs)

 def _open_gripper_for_box_release(self,target_object):
  if target_object=='beverage_bottle':
   position=np.asarray(self._current_object_position(target_object),float)
   if not np.isfinite(position).all() or not .115<=position[2]<=.130:
    raise GuardViolation(f'瓶子释放高度未进入115–130毫米区间：z={position[2]:.6f}m')
   if np.linalg.norm(position[:2]-np.asarray(self.box_center))>.020:
    raise GuardViolation('瓶子释放前蓝盒水平误差超过20毫米')
  return super()._open_gripper_for_box_release(target_object)
