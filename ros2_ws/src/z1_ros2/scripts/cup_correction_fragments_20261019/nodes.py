#!/usr/bin/python3
"""Cup-only collection subclasses; production nodes are never edited."""
import copy,json,os,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from core import atomic_json
from fragment_core import valid_hold,partial_labels
from geometry import Geometry
from preflight import routes,choose_route,trajectory_points

def context():
 return json.loads(Path(os.environ['Z1_CORRECTION_CURRENT']).read_text())
def evidence_path(ctx):
 return Path(os.environ['Z1_CORRECTION_CURRENT']).parent/('evidence_'+ctx['attempt_id']+'.json')

def assistant_main():
 import z1_isaac_bridge.grasp_assistant as module
 module.GRASP_PROFILES=copy.deepcopy(module.GRASP_PROFILES)
 module.OBJECT_GRASP_PROFILES=copy.deepcopy(module.OBJECT_GRASP_PROFILES)
 module.OBJECT_GRASP_PROFILES['cup']['physical_test_lift_minimum_accepted_z_m']=.05
 module.GRASP_PROFILES['cup'].update(transport_object_z_m=.06,physical_test_lift_safe_completion_z_m=.05,
    minimum_object_rise_m=.03,physical_test_lift_completion_tolerance_m=.005)
 Base=module.GraspAssistant
 class CupCorrectionAssistant(Base):
  def __init__(self):
   super().__init__();self.geometry=Geometry();self.cached=None;self.selecting_reference=False;self.box_approach_object_z=.06
  def _generate_plan(self):
   from moveit_msgs.msg import RobotState
   self.cached=None;self.selecting_reference=True
   try:message=super()._generate_plan()
   finally:self.selecting_reference=False
   with self._lock:
    initial=[self._joint_positions[f'joint{i}'] for i in range(1,7)]+[self.gripper_open]
    approved=copy.deepcopy(self._approved_trajectory)
   # FK of the actual planned endpoint, not a bottle grasp-height heuristic.
   end=trajectory_points(approved.joint_trajectory,initial)[-1]
   p,r=self.geometry.link_fk(end);q=Rotation.from_matrix(r).as_quat().tolist()
   state=RobotState();state.is_diff=True
   state.joint_state.name=[f'joint{i}' for i in range(1,7)]+['jointGripper'];state.joint_state.position=initial
   deadline=time.monotonic()+75
   def plan_route(route):
    current=copy.deepcopy(state);planned=[]
    for step in route['preparation']+route['recovery']:
     self._raise_if_stop_requested()
     if time.monotonic()>deadline:raise RuntimeError('纠偏路径规划预算耗尽')
     seed=dict(zip(current.joint_state.name,current.joint_state.position))
     start=[seed[f'joint{i}'] for i in range(1,7)]+[self.gripper_open]
     result=self._plan_candidate(dict(position=step['position'],orientation=step['orientation']),
        linear=step['linear'],start_state=current,open_gripper=True,position_tolerance=.002,
        orientation_tolerance=.025,velocity_scale=.10,acceleration_scale=.10)
     pts=trajectory_points(result.planned_trajectory.joint_trajectory,start)
     self.geometry.check_path(pts)
     planned.append((step,result.planned_trajectory,start))
     current=self._state_after_trajectory(result)
    return planned
   route,planned,rejected=choose_route(routes(p,q,self._current_object_position('cup'),context()['task']),plan_route)
   self.cached=(route,planned,rejected);self.reference=(p.tolist(),q)
   return message+'；杯子纠偏准备与恢复路径已预检'
  def execute_cached(self,item):
   step,trajectory,start=item;self._raise_if_stop_requested()
   # Use the existing controller's tracking/stop handling. No extra 5/6 mm
   # Cartesian tracking failure threshold is introduced here.
   if self._send_arm_trajectory(trajectory):raise RuntimeError('杯子纠偏动作已停止')
   return self._current_ee_pose()
  def _execute_plan(self):
   if self.cached is None:raise RuntimeError('缺少杯子纠偏完整预检')
   ctx=context();self.ctx=ctx;route,planned,rejected=self.cached
   self._set_bottle_finger_collision(False)
   self._execute_gripper('open')
   self._command_scene_grasp('unfreeze','cup')
   if not self._wait_scene_grasp('UNFROZEN','',5):raise RuntimeError('杯子未恢复动态物理')
   initial_object=self._current_object_position('cup')
   for item in planned[:len(route['preparation'])]:self.execute_cached(item)
   # Establish a true open command before recording; actual-angle holds can
   # otherwise exceed the close-label threshold without a close request.
   self._execute_gripper('open')
   actual_p,actual_q=self._current_ee_pose()
   with self._lock:state=[self._joint_positions[f'joint{i}'] for i in range(1,7)]+[self._joint_positions['jointGripper']]
   # Reject preparation that pushed/rolled the fruit, as in the bottle
   # collector; this is sample validity, not a target-tracking threshold.
   shift=float(np.linalg.norm(np.array(self._current_object_position('cup'))-initial_object))
   if shift>.005:raise RuntimeError(f'准备偏差时杯子被推移{shift:.4f}m，本片段不计入纠偏配额')
   task=ctx['task']
   meta=dict(task_id=task['id'],target_object='cup',task=task,plan_hash=ctx['plan_hash'],attempt_id=ctx['attempt_id'],
      collector_code_hash=ctx.get('collector_code_hash'),accepted=False,recovery_complete=False,
      reference_position=self.reference[0],reference_orientation=self.reference[1],
      requested_position=route['bad_position'],requested_orientation=route['bad_orientation'],actual_position=list(actual_p),actual_orientation=list(actual_q),
      preflight=dict(route=route,rejected_routes=rejected,complete_segments=len(planned)),start_joint_state=state,
      initial_cup_position_m=initial_object,actual_cup_position_m=self._current_object_position('cup'),
      minimum_lift_m=.05,lift_target_m=.06,stages=[])
   atomic_json(evidence_path(ctx),meta)
   with self._lock:self._approved_trajectory=None;self._approved_target=''
   return '杯子偏离状态已准备；下一步开始纠正短片段'
  def _guided_grasp(self):
   ctx=self.ctx;meta=json.loads(evidence_path(ctx).read_text());route,planned,_=self.cached
   moving=[]
   for item in planned[len(route['preparation']):]:
    step=item[0];begin=self.get_clock().now().nanoseconds/1e9
    p,q=self.execute_cached(item)
    with self._lock:moving.append([self._joint_positions[f'joint{i}'] for i in range(1,7)])
    meta['stages'].append(dict(stage=step['stage'],start_sim_time=begin,end_sim_time=self.get_clock().now().nanoseconds/1e9,
           actual_position=list(p),actual_orientation=list(q)))
    atomic_json(evidence_path(ctx),meta)
   meta['recovery_complete']=True
   if moving:
    meta['joint_ranges_rad']=np.ptp(np.array(moving),axis=0).tolist()
   atomic_json(evidence_path(ctx),meta)
   # Keep cup-specific preclose geometry, closing, preload and physical proof.
   return super()._guided_grasp()
  def _carry_loaded_object_toward_box(self,target_object,envelope,minimum_object_z,stage,require_box_approach=False,preferred_object_z=None):
   origin=self._current_object_position('cup');p,q=self._current_ee_pose()
   target=(p[0],p[1],p[2]+max(0.,.06-origin[2]))
   self._raise_if_stop_requested()
   error=None
   for linear in (True,False):
    try:
     result=self._plan_candidate(dict(position=target,orientation=q),linear=linear,position_tolerance=.002,
          orientation_tolerance=.04,velocity_scale=self.physical_test_lift_velocity_scale,
          acceleration_scale=self.physical_test_lift_velocity_scale)
     error=None;break
    except RuntimeError as exc:
     if not module._is_recoverable_lift_planning_error(exc):raise
     error=exc
   if error:raise error
   with self._lock:initial=[self._joint_positions[f'joint{i}'] for i in range(1,7)]+[self._joint_positions['jointGripper']]
   self.geometry.check_path(trajectory_points(result.planned_trajectory.joint_trajectory,initial))
   if self._send_arm_trajectory(result.planned_trajectory):raise RuntimeError('杯子纠偏试抬停止')
   return self._require_loaded_physical_grasp('cup',envelope,'杯子纠偏离桌后',use_verified_lift_reference=False,
      loaded_envelope_margin_m=self.transport_loaded_grasp_guard_margin)
 module.GraspAssistant=CupCorrectionAssistant;module.main()

def gui_main():
 import z1_isaac_bridge.grasp_gui_bridge as module
 module.GRID_LAYOUT_RESAMPLE_ATTEMPTS=2
 Base=module.GraspGuiBridge
 class CupCorrectionGui(Base):
  def auto_demonstration(self,*args,**kwargs):
   try:return super().auto_demonstration(*args,**kwargs)
   except Exception as exc:
    ctx=context();atomic_json(evidence_path(ctx).with_suffix('.failure.json'),dict(context=ctx,error=str(exc),status=self.snapshot()));raise
  def start_recording(self):
   return {'success':True,'message':'杯子纠偏将在偏离状态准备完成后开始录制'}
  def _execute_pregrasp_for_auto(self):
   result=super()._execute_pregrasp_for_auto()
   Base.start_recording(self)
   deadline=time.monotonic()+15
   while True:
    with self._lock:frames=int(self._recording_status.get('frame_count',0))
    if frames>=3:break
    if self._force_stop_event.is_set():raise RuntimeError('录制起始张爪帧时停止')
    if time.monotonic()>deadline:raise RuntimeError('等待三帧张爪观测超时')
    time.sleep(.05)
   return result
  def _guided_place_for_auto(self):
   start=self.get_clock().now().nanoseconds/1e9;deadline=time.monotonic()+30
   while self.get_clock().now().nanoseconds/1e9-start<.6:
    if self._force_stop_event.is_set():raise RuntimeError('杯子纠偏保持已停止')
    if time.monotonic()>deadline:raise RuntimeError('杯子纠偏保持时钟超时')
    with self._lock:status=dict(self._status)
    if not valid_hold(status):raise RuntimeError('杯子纠偏保持未通过真实双指夹持和离桌检查')
    time.sleep(.05)
   return {'success':True,'message':'杯子纠正、夹取、抬升及0.6秒保持完成；无放置动作'}
 module.GraspGuiBridge=CupCorrectionGui;module.main()

def recorder_main():
 import z1_isaac_bridge.episode_recorder as module
 Base=module.EpisodeRecorder
 class CupCorrectionRecorder(Base):
  def __init__(self):self.contact=False;super().__init__()
  def _task_status_callback(self,msg):
   super()._task_status_callback(msg)
   with self._lock:
    if self._recording:self.contact |= bool(self._task_status.get('left_finger_touching_table') or self._task_status.get('right_finger_touching_table'))
  def _start_callback(self,req,res):
   ctx=context();meta=json.loads(evidence_path(ctx).read_text());self.contact=False;self.require_task_success=False
   if meta['task_id']!=ctx['task']['id']:raise RuntimeError('杯子纠偏任务证据不匹配')
   res=super()._start_callback(req,res)
   if res.success:
    self.ctx=ctx
    with self._lock:atomic_json(self._episode_dir/'correction.json',meta)
   return res
  def _save_callback(self,req,res):
   ctx=self.ctx;meta=json.loads(evidence_path(ctx).read_text())
   with self._lock:
    if self.contact or not meta.get('recovery_complete') or not valid_hold(self._task_status):
     res.success=False;res.message='杯子纠偏未通过恢复完成、无碰桌及物理夹持验收';return res
    self._frames_file.flush()
    rows=[json.loads(l) for l in (self._episode_dir/'frames.jsonl').read_text().splitlines()]
    try:labels,onset,last=partial_labels(rows)
    except ValueError as exc:
     atomic_json(evidence_path(ctx).with_suffix('.label_failure.json'),dict(error=str(exc),actions=[r['action'][-1] for r in rows]))
     res.success=False;res.message=str(exc);return res
    meta.update(accepted=True,split=ctx['task']['split'],fragment_goal='cup_correction_grasp_lift',final_status=dict(self._task_status),
      close_command_frame=onset,valid_chunk16_last_anchor=last,table_contact=False,
      phase_spans=meta.get('stages',[]),gripper_label_rule='first close command > -0.895rad, no GRASPING phase requirement')
    atomic_json(self._episode_dir/'correction.json',meta)
    atomic_json(self._episode_dir/'gripper_labels.json',dict(labels=labels,close_command_frame=onset,raw_state_encoding='actual_joint_angles'))
    index=int(self._episode_index)
    # The reentrant lock prevents a timer appending an unlabeled final frame.
    res=super()._save_callback(req,res)
    if res.success:
     root=self.dataset_root/f'episode_{index:06d}';summary=json.loads((root/'episode_summary.json').read_text())
     summary.update(episode_goal='cup_correction_grasp_lift',fragment_success=True,split=ctx['task']['split'],task_id=ctx['task']['id'])
     atomic_json(root/'episode_summary.json',summary)
     temp=self.dataset_root/'episodes.jsonl.tmp'
     with temp.open('w') as f:
      for ep in sorted(self.dataset_root.glob('episode_[0-9]*')):f.write(json.dumps(json.loads((ep/'episode_summary.json').read_text()),ensure_ascii=False)+'\n')
      f.flush();os.fsync(f.fileno())
     os.replace(temp,self.dataset_root/'episodes.jsonl')
    return res
 module.EpisodeRecorder=CupCorrectionRecorder;module.main()

if __name__=='__main__':
 role=sys.argv.pop(1)
 {'assistant':assistant_main,'gui':gui_main,'recorder':recorder_main}[role]()
