"""Plan complete loaded carry and release descent before first loaded motion."""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from moveit_msgs.msg import RobotState
from z1_isaac_bridge.grasp_config import OBJECT_GRASP_PROFILES
LOG=Path('/home/lp531/z1_queue_logs/bottle_approach_height30_20261112')

class BottleLookahead:
 def __init__(self,*args,**kwargs):
  self._bottle_cached_descent=None
  super().__init__(*args,**kwargs)
 def _reset_assistant_service(self,*args):
  self._bottle_cached_descent=None
  return super()._reset_assistant_service(*args)
 def _stop_assistant_service(self,*args):
  self._bottle_cached_descent=None
  return super()._stop_assistant_service(*args)
 def _lookahead_state(self,q,gripper):
  state=RobotState();state.is_diff=True;state.joint_state.name=[f'joint{i}' for i in range(1,7)]+['jointGripper'];state.joint_state.position=[float(v) for v in q]+[float(gripper)];state.joint_state.velocity=[0.]*7
  return state
 def _plan_bottle_route(self,origin):
  with self._lock:q0=np.array([self._joint_positions[f'joint{i}'] for i in range(1,7)]);gripper=self._joint_positions['jointGripper']
  ee,r0=self.geometry.link_fk(q0);offset=r0.T@(origin-ee);errors=[];started=time.monotonic()
  # Preserve direct carry plus original diagonal-midpoint fallback, with continuous attitudes.
  for midpoint in (False,True):
   for tolerance in (.2,.4):
    q=q0.copy();rotation=r0.copy();route=[]
    goals=[]
    if midpoint:goals.append(('transition',[(origin[0]+self.box_center[0])/2,(origin[1]+self.box_center[1])/2,.190]))
    goals.extend([('carry',[*self.box_center,.190]),('descent',[*self.box_center,float(self.box_release_object_z)])])
    try:
     for phase,goal in goals:
      self._raise_if_stop_requested();self._set_status('JOG_PLANNING',f'闭爪后提前规划完整搬运和下降：{phase}',False)
      candidate=dict(position=goal,orientation=Rotation.from_matrix(rotation).as_quat().tolist(),target_point_offset=offset.tolist())
      if phase!='descent':candidate['position_z_range']=[.180,.200]
      else:
       candidate['position']=(np.asarray(goal)-rotation@offset).tolist()
       candidate.pop('target_point_offset')
      plan=self._plan_candidate(candidate,linear=(phase=='descent'),position_tolerance=.002,orientation_tolerance=(.04 if phase=='descent' else tolerance),velocity_scale=.08,acceleration_scale=.04,start_state=self._lookahead_state(q,gripper))
      trajectory=plan.planned_trajectory;jt=trajectory.joint_trajectory
      first=np.array([jt.points[0].positions[jt.joint_names.index(f'joint{i}')] for i in range(1,7)])
      if np.max(np.abs(first-q))>.01:raise RuntimeError('预规划轨迹起点与上一段终点不一致')
      _,delta=self._trajectory_score(jt)
      allowed_delta=(float(self.max_joint_delta) if phase=='descent' else float(OBJECT_GRASP_PROFILES['beverage_bottle'].get('loaded_transport_max_joint_delta_rad',self.max_joint_delta)))
      if delta>allowed_delta:raise RuntimeError('预规划轨迹超过原关节变化限制')
      end=np.array([jt.points[-1].positions[jt.joint_names.index(f'joint{i}')] for i in range(1,7)])
      p,r=self.geometry.link_fk(end);obj=p+r@offset
      if p[2]<self.manual_minimum_z or p[2]>.75 or np.linalg.norm(p[:2])>.78:raise RuntimeError('预规划终点超过原工作空间')
      if np.linalg.norm(obj[:2]-np.asarray(goal[:2]))>.003:raise RuntimeError('预规划瓶子终点水平位置超差')
      if phase!='descent' and not .180<=obj[2]<=.200:raise RuntimeError('预规划搬运高度未进入180–200毫米')
      if phase=='descent' and abs(obj[2]-goal[2])>.003:raise RuntimeError('预规划释放高度超差')
      start_ee,start_rotation=self.geometry.link_fk(q);start_object=start_ee+start_rotation@offset
      route.append(dict(start_object=start_object,phase=phase,trajectory=trajectory,start_q=q.copy(),end_q=end,object_goal=obj.tolist(),offset=offset.copy()))
      q=end;rotation=r
     if np.linalg.norm(np.asarray(self._current_object_position('beverage_bottle'))-origin)>.003:raise RuntimeError('预规划期间瓶子移动，已放弃本次路线')
     with self._lock:actual=np.array([self._joint_positions[f'joint{i}'] for i in range(1,7)])
     if np.max(np.abs(actual-q0))>.01:raise RuntimeError('预规划期间机械臂移动，已放弃本次路线')
     LOG.mkdir(parents=True,exist_ok=True);path=LOG/f'lookahead_{time.time_ns()}.json';path.write_text(json.dumps(dict(all_planned_before_loaded_motion=True,planning_duration_s=time.monotonic()-started,midpoint=midpoint,orientation_tolerance_rad=tolerance,legs=[dict(phase=l['phase'],object_goal=l['object_goal'],start_q=l['start_q'].tolist(),end_q=l['end_q'].tolist()) for l in route]),ensure_ascii=False,indent=2))
     self.get_logger().info('完整搬运及下降已提前规划，执行途中不再规划：'+str(path));return route
    except (RuntimeError,ValueError) as exc:
     errors.append(str(exc));self.get_logger().warning('完整路线候选被拒绝（尚未带载移动）：'+str(exc))
  raise RuntimeError('完整搬运及下降预规划失败（尚未带载移动）：'+' | '.join(errors))
 def _execute_bottle_leg(self,leg):
  self._raise_if_stop_requested()
  with self._lock:q=np.array([self._joint_positions[f'joint{i}'] for i in range(1,7)])
  if np.max(np.abs(q-leg['start_q']))>.01:raise RuntimeError('已预规划轨迹起点发生偏移，取消本次且不现场重规划')
  if np.linalg.norm(np.asarray(self._current_object_position('beverage_bottle'))-leg['start_object'])>.003:raise RuntimeError('已预规划瓶子起点发生偏移，取消本次且不现场重规划')
  self._set_status('JOGGING',f'执行已提前规划的瓶子轨迹：{leg["phase"]}',False)
  if self._send_arm_trajectory(leg['trajectory']):raise RuntimeError('已预规划运动被停止')
 def _carry_loaded_object_toward_box(self,target_object,loaded_envelope,minimum_object_z,stage,require_box_approach=False,preferred_object_z=None):
  if target_object!='beverage_bottle':return super()._carry_loaded_object_toward_box(target_object,loaded_envelope,minimum_object_z,stage,require_box_approach,preferred_object_z)
  def verify(label):return self._require_loaded_physical_grasp(target_object,loaded_envelope,label,use_verified_lift_reference=False,loaded_envelope_margin_m=self.transport_loaded_grasp_guard_margin)
  position=np.asarray(self._current_object_position(target_object))
  if getattr(self,'_bottle_cached_descent',None) is not None:
   if .180<=position[2]<=.200 and np.linalg.norm(position[:2]-np.asarray(self.box_center))<=.020:return verify(stage)
   self._bottle_cached_descent=None;raise RuntimeError('已规划搬运完成后位姿偏移，取消本次且不追加规划')
  verify('完整路线预规划前');route=self._plan_bottle_route(np.asarray(self._current_object_position(target_object)));verify('完整路线执行前')
  try:
   for leg in route[:-1]:
    self._execute_bottle_leg(leg);verify('已规划'+leg['phase']+'执行后')
   position=np.asarray(self._current_object_position(target_object))
   if not .180<=position[2]<=.200 or np.linalg.norm(position[:2]-np.asarray(self.box_center))>.020:raise RuntimeError('已预规划搬运未进入目标高度或水平范围')
   self._bottle_cached_descent=route[-1];return verify(stage)
  except BaseException:self._bottle_cached_descent=None;raise
 def _move_object_to(self,destination,description,linear,*args,**kwargs):
  if self.target_object=='beverage_bottle' and description=='垂直放入蓝盒':
   leg=getattr(self,'_bottle_cached_descent',None)
   if leg is None:raise RuntimeError('没有有效的已预规划下降轨迹，拒绝现场追加规划')
   self._bottle_cached_descent=None
   position=np.asarray(self._current_object_position('beverage_bottle'))
   p,r=self.geometry.link_fk([self._joint_positions[f'joint{i}'] for i in range(1,7)])
   if np.linalg.norm(r.T@(position-p)-leg['offset'])>getattr(self,'physical_test_lift_relative_position_tolerance',.04):raise RuntimeError('下降前夹持偏置变化超出原滑移限制')
   if np.linalg.norm(position[:2]-np.asarray(self.box_center))>.020:raise RuntimeError('下降前未保持蓝盒水平对中')
   self._execute_bottle_leg(leg);return
  if self.target_object=='beverage_bottle' and '水平对中' in description and getattr(self,'_bottle_cached_descent',None) is not None:
   raise RuntimeError('已规划路线执行后对中超差，取消本次且不现场追加规划')
  return super()._move_object_to(destination,description,linear,*args,**kwargs)
