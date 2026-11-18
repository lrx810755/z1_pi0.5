#!/usr/bin/python3
"""Run isolated subclasses of the existing collection nodes."""
import json
import copy
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'z1_isaac_bridge'))
from core import atomic_json,recovery_waypoints,strict_final,alignment_step
from fragment_core import valid_hold
from geometry import Geometry
from preflight import routes,choose_route,trajectory_points

def context():
    return json.loads(Path(os.environ['Z1_CORRECTION_CURRENT']).read_text())

def evidence_path(ctx):
    return Path(os.environ['Z1_CORRECTION_CURRENT']).parent/('evidence_'+ctx['attempt_id']+'.json')

def assistant_main():
    import z1_isaac_bridge.grasp_assistant as module
    module.GRASP_PROFILES=copy.deepcopy(module.GRASP_PROFILES)
    module.GRASP_PROFILES["beverage_bottle"]["single_pass_box_alignment"]=False
    module.GRASP_PROFILES['beverage_bottle'].update(transport_object_z_m=.06,
        physical_test_lift_safe_completion_z_m=.05,minimum_object_rise_m=.03,
        physical_test_lift_completion_tolerance_m=.005)
    module.BOX_ALIGNMENT_MAX_STEPS=10
    module.BOX_RELEASE_XY_TOLERANCE_M["beverage_bottle"]=.008
    Base=module.GraspAssistant
    class CorrectionAssistant(Base):
        def __init__(self):
            super().__init__();self.correction_geometry=Geometry();self.correction_guard=False;self.box_approach_object_z=.06

        def _carry_loaded_object_toward_box(self,target_object,loaded_envelope,minimum_object_z,stage,require_box_approach=False,preferred_object_z=None):
            # Fragment teacher ends at an in-place physical lift. Keep the
            # original height/contact/slip proof; avoid unrelated box carry.
            goal=max(float(minimum_object_z),float(preferred_object_z or minimum_object_z))
            for attempt in range(30):
                self._raise_if_stop_requested()
                obj=self._current_object_position(target_object)
                if obj[2]>=goal-.002:break
                step=min(.020,goal-obj[2])
                position,orientation=self._current_ee_pose()
                target=(position[0],position[1],position[2]+step)
                error=None
                for linear in [True,False]:
                    try:
                        self._execute_manual_pose(target,orientation,'纠偏片段原位物理抬升',linear=linear,
                            max_joint_delta=.35,position_tolerance=.002,orientation_tolerance=.025,
                            velocity_scale=self.physical_test_lift_velocity_scale)
                        error=None;break
                    except RuntimeError as exc:
                        error=exc
                        if not module._is_recoverable_lift_planning_error(exc):raise
                if error:raise RuntimeError('原位物理抬升规划失败: '+str(error))
                if self._task_flag('left_finger_touching_table') or self._task_flag('right_finger_touching_table'):
                    raise RuntimeError('原位物理抬升碰桌')
                time.sleep(.25)
                self._require_loaded_physical_grasp(target_object,loaded_envelope,'纠偏片段分段原位抬升后',
                    use_verified_lift_reference=False,loaded_envelope_margin_m=self.transport_loaded_grasp_guard_margin)
                if self._current_object_position(target_object)[2]-obj[2]<.001:
                    if self._current_object_position(target_object)[2]>=minimum_object_z:break
                    raise RuntimeError(f'原位物理抬升无有效进展: before={obj[2]:.4f}, after={self._current_object_position(target_object)[2]:.4f}, minimum={minimum_object_z:.4f}, goal={goal:.4f}')
            else:raise RuntimeError('原位物理抬升超过迭代上限')
            if self._current_object_position(target_object)[2]<minimum_object_z:
                raise RuntimeError('原位物理抬升未达到原老师的验收高度')
            return self._require_loaded_physical_grasp(target_object,loaded_envelope,'纠偏片段原位抬升后',
                use_verified_lift_reference=False,loaded_envelope_margin_m=self.transport_loaded_grasp_guard_margin)

        def _generate_plan(self):
            from moveit_msgs.msg import RobotState
            self.selecting_correction_reference=True
            try:message=super()._generate_plan()
            except RuntimeError as exc:
                self.get_logger().error("BASE_PREFLIGHT_REJECT: "+str(exc));raise
            finally:self.selecting_correction_reference=False
            self.correction_cached=None
            with self._lock:
                initial=[self._joint_positions[f'joint{i}'] for i in range(1,7)]+[self.gripper_open]
                approved=copy.deepcopy(self._approved_trajectory)
            points=trajectory_points(approved.joint_trajectory,initial)
            self.correction_geometry.check_path(points)
            end=points[-1]
            p,r=self.correction_geometry.link_fk(end)
            q=Rotation.from_matrix(r).as_quat().tolist()
            state=RobotState();state.is_diff=True
            state.joint_state.name=[f'joint{i}' for i in range(1,7)]+['jointGripper']
            state.joint_state.position=initial
            deadline=time.monotonic()+75
            def plan_route(route):
                current=copy.deepcopy(state);planned=[]
                for step in route['preparation']+route['recovery']:
                    self._raise_if_stop_requested()
                    if time.monotonic()>deadline:raise RuntimeError('全路径预检时间预算耗尽')
                    target=step['position']
                    if target[2]<self.manual_minimum_z or target[2]>.75 or np.linalg.norm(target)>.78:
                        raise RuntimeError(step['stage']+': 目标超出工作空间')
                    seed=dict(zip(current.joint_state.name,current.joint_state.position))
                    start=[seed[f'joint{i}'] for i in range(1,7)]+[self.gripper_open]
                    kwargs=dict(position_tolerance=.001,orientation_tolerance=.025,
                                velocity_scale=.10,acceleration_scale=.10,start_state=current,open_gripper=True)
                    candidate=dict(position=tuple(target),orientation=tuple(step['orientation']))
                    try:result=self._plan_candidate(candidate,linear=step['linear'],**kwargs)
                    except RuntimeError as first_error:
                        if not step['linear'] or time.monotonic()>deadline:
                            raise RuntimeError(step['stage']+': '+str(first_error)) from first_error
                        try:result=self._plan_candidate(candidate,linear=False,**kwargs)
                        except RuntimeError as exc:
                            self.get_logger().warning(f"ROUTE_REJECT stage={step['stage']} target={target} start={start} error={exc}")
                            raise RuntimeError(step['stage']+': '+str(exc)) from exc
                    pts=trajectory_points(result.planned_trajectory.joint_trajectory,start)
                    if step['stage']!='prepare_lift' and np.max(np.abs(np.array(pts)[:,:6]-np.array(start[:6])))>.7:
                        raise RuntimeError(step['stage']+': 关节运动幅度超过限制')
                    self.correction_geometry.check_path(pts)
                    planned.append((step,result.planned_trajectory,start))
                    current=self._state_after_trajectory(result)
                return planned
            try:
                route,planned,rejections=choose_route(routes(p,q,self._current_object_position('beverage_bottle'),context()['task']),plan_route)
            except RuntimeError as exc:
                self.get_logger().error("CORRECTION_PREFLIGHT_REJECT: "+str(exc))
                with self._lock:self._approved_trajectory=None
                raise
            self.correction_cached=(route,planned,rejections)
            self.correction_reference=(p.tolist(),q)
            self.get_logger().info(f'纠正全路径预检通过：8 段，退让 {route["retreat_m"]:.3f} m，抬高 {route["lift_m"]:.3f} m')
            return message+'；纠正准备与恢复全路径预检通过'

        def correction_execute_cached(self,item):
            step,trajectory,start=item
            self._raise_if_stop_requested()
            with self._lock:actual=[self._joint_positions[f'joint{i}'] for i in range(1,7)]
            if np.max(np.abs(np.array(actual)-start[:6]))>.04:
                raise RuntimeError(step['stage']+': 实际关节状态偏离预检起点')
            self.get_logger().info('Correction cached stage: '+step['stage'])
            if self._send_arm_trajectory(trajectory):raise RuntimeError('纠正轨迹被停止')
            p,q=self._current_ee_pose()
            if np.linalg.norm(np.array(p)-step['position'])>.005 or (Rotation.from_quat(q).inv()*Rotation.from_quat(step['orientation'])).magnitude()>np.deg2rad(3):
                raise RuntimeError(step['stage']+': 预检轨迹实际跟踪超限')
            if self._task_flag('left_finger_touching_table') or self._task_flag('right_finger_touching_table'):
                raise RuntimeError('纠正动作触桌')
            return list(p),list(q)

        def _send_arm_trajectory(self,trajectory):
            if self.correction_guard:
                with self._lock:
                    initial=[self._joint_positions[f'joint{i}'] for i in range(1,7)]+[self._joint_positions['jointGripper']]
                points=[initial]
                for point in trajectory.joint_trajectory.points:
                    q=list(initial)
                    for name,value in zip(trajectory.joint_trajectory.joint_names,point.positions):
                        if name in [f'joint{i}' for i in range(1,7)]:q[int(name[-1])-1]=value
                    points.append(q)
                self.correction_geometry.check_path(points)
                self.correction_motion_points.extend(points)
            return super()._send_arm_trajectory(trajectory)

        def correction_move(self,p,q,label,linear=True):
            self.get_logger().info('Correction stage: '+label)
            try:
                self._execute_manual_pose(tuple(p),tuple(q),label,linear=linear,
                    position_tolerance=.001,orientation_tolerance=.025,max_joint_delta=.7,velocity_scale=.10,acceleration_scale=.10)
            except RuntimeError as exc:
                # A failed LIN IK can still have a collision-free joint-space path.
                # The same dense finger-height check also guards this fallback.
                if not linear or 'MoveIt error_code' not in str(exc):raise RuntimeError(label+': '+str(exc)) from exc
                self._execute_manual_pose(tuple(p),tuple(q),label+'（关节空间规划）',linear=False,
                    position_tolerance=.001,orientation_tolerance=.025,max_joint_delta=.7,velocity_scale=.10,acceleration_scale=.10)
            actual_p,actual_q=self._current_ee_pose()
            position_error=float(np.linalg.norm(np.array(actual_p)-p))
            angle=float((Rotation.from_quat(q).inv()*Rotation.from_quat(actual_q)).magnitude())
            if position_error>.005 or angle>np.deg2rad(3):
                raise RuntimeError(f'纠正位姿未到达: {position_error:.4f} m, {np.rad2deg(angle):.2f} deg')
            if self._task_flag('left_finger_touching_table') or self._task_flag('right_finger_touching_table'):
                raise RuntimeError('纠正动作触桌')
            return list(actual_p),list(actual_q)

        def _execute_plan(self):
            ctx=context();self.correction_ctx=ctx
            if self.correction_cached is None:raise RuntimeError('缺少纠正全路径预检，禁止移动')
            route,planned,rejections=self.correction_cached
            self._raise_if_stop_requested()
            self._set_status('PREOPEN','从安全起点直接准备纠偏状态',False)
            self._execute_gripper('open')
            self._command_scene_grasp('unfreeze')
            if not self._wait_scene_grasp('UNFROZEN','',5):raise RuntimeError('准备目标未解除固定')
            message='已从安全起点直接准备纠偏状态'
            measured_p,measured_q=self._current_ee_pose()
            p,q=self.correction_reference
            self.correction_motion_points=[];self.correction_guard=True
            try:
                object_p=np.array(self._current_object_position('beverage_bottle'))
                task=ctx['task'];anchor=np.array(route['anchor_position'])
                bad=np.array(route['bad_position']);bad_q=route['bad_orientation']
                for item in planned[:4]:
                    actual_p,actual_q=self.correction_execute_cached(item)
                    self.get_logger().info(f"PREP_OBJECT stage={item[0]['stage']} start={object_p.tolist()} current={self._current_object_position('beverage_bottle')}")
                self._command_scene_grasp('unfreeze')
                if not self._wait_scene_grasp('UNFROZEN','',5):raise RuntimeError('目标物未解除固定')
                time.sleep(.25)
                final_object_p=np.array(self._current_object_position('beverage_bottle'))
                object_shift=float(np.linalg.norm(final_object_p-object_p))
                if object_shift>.005:
                    raise RuntimeError(f'准备偏离状态时瓶子发生位移 {object_shift:.4f} m，start={object_p.tolist()}, end={final_object_p.tolist()}，拒绝记录')
                with self._lock:
                    state=[self._joint_positions[f'joint{i}'] for i in range(1,7)]+[self._joint_positions['jointGripper']]
                meta=dict(task_id=task['id'],plan_hash=ctx['plan_hash'],attempt_id=ctx['attempt_id'],task=task,
                    collector_code_hash=ctx.get('collector_code_hash'),
                    preflight=dict(route=route,rejected_routes=rejections,complete_segments=len(planned)),
                    reference_position=p,reference_orientation=q,anchor_position=anchor.tolist(),
                    preparation_start_position=measured_p,preparation_start_orientation=measured_q,
                    requested_position=bad.tolist(),requested_orientation=bad_q,actual_position=actual_p,actual_orientation=actual_q,
                    actual_offset_world_from_anchor_m=(np.array(actual_p)-anchor).tolist(),
                    finger_height_m=self.correction_geometry.height(state),start_joint_state=state,
                    accepted=False,recovery_complete=False,fragment_lift_target_m=.06,fragment_minimum_lift_m=.05,stages=[])
                atomic_json(evidence_path(ctx),meta)
            finally:self.correction_guard=False
            with self._lock:self._approved_trajectory=None;self._approved_target=''
            self._set_status('HANDED_OVER','纠正偏离状态已就绪',False)
            return message+'；纠正偏离状态已就绪'

        def _guided_grasp(self):
            ctx=self.correction_ctx;meta=json.loads(evidence_path(ctx).read_text())
            p,q=self._current_ee_pose();refp,refq=self.correction_reference
            self.correction_motion_points=[];self.correction_guard=True
            try:
                for item in ([] if ctx['task']['kind']=='close_ready' else self.correction_cached[1][4:]):
                    step=item[0]
                    start=self.get_clock().now().nanoseconds/1e9
                    actual_p,actual_q=self.correction_execute_cached(item)
                    meta['stages'].append(dict(**step,start_sim_time=start,end_sim_time=self.get_clock().now().nanoseconds/1e9,
                                              actual_position=actual_p,actual_orientation=actual_q))
                    atomic_json(evidence_path(ctx),meta)
                # The original one-pass teacher accepts <=6 mm residuals.
                # Correct measured XYZ error while open and dynamic, rather
                # than freezing the object or relaxing the acceptance band.
                desired=self._grasp_candidate['desired_object_in_stator_m']
                for attempt in range(5):
                    self._command_scene_grasp('probe','beverage_bottle')
                    if not self._wait_scene_grasp('PROBED','beverage_bottle',5):raise RuntimeError('精对齐缺少物体相对位置')
                    with self._lock:local=self._scene_grasp_status.get('local_position_m',[])
                    if len(local)!=3:raise RuntimeError('精对齐相对位置无效')
                    error=np.array(local)-desired
                    capture=module.OBJECT_GRASP_PROFILES['beverage_bottle']['preclose_capture_envelope']
                    if np.max(np.abs(error))<=.004 and module.inside_grasp_envelope(local,capture):break
                    if attempt==4:raise RuntimeError('动态开爪精对齐未收敛')
                    p,q=self._current_ee_pose()
                    shift=module.rotate_vector(self._current_stator_orientation(),tuple(alignment_step(local,desired)))
                    start=self.get_clock().now().nanoseconds/1e9
                    ap,aq=self.correction_move((np.array(p)+shift).tolist(),q,'fine_align',True)
                    meta['stages'].append(dict(stage='fine_align',start_sim_time=start,end_sim_time=self.get_clock().now().nanoseconds/1e9,
                                              measured_local=local,desired_local=list(desired),actual_position=ap,actual_orientation=aq))
                    atomic_json(evidence_path(ctx),meta)
                spans=np.ptp(np.array(self.correction_motion_points)[:,:6],axis=0) if self.correction_motion_points else np.zeros(6)
                meta['moving_arm_joints']=int(np.sum(spans>.003));meta['joint_ranges_rad']=spans.tolist()
                if meta['moving_arm_joints']<2 and ctx['task']['kind']!='close_ready':raise RuntimeError('纠正轨迹未体现多关节运动')
                meta['recovery_complete']=True;atomic_json(evidence_path(ctx),meta)
            finally:self.correction_guard=False
            return super()._guided_grasp()
    module.GraspAssistant=CorrectionAssistant;module.main()

def gui_main():
    import z1_isaac_bridge.grasp_gui_bridge as module
    module.GRID_LAYOUT_RESAMPLE_ATTEMPTS=2
    Base=module.GraspGuiBridge
    class CorrectionGui(Base):
        def auto_demonstration(self,*args,**kwargs):
            try:return super().auto_demonstration(*args,**kwargs)
            except Exception as exc:
                ctx=context()
                atomic_json(evidence_path(ctx).with_suffix('.failure.json'),dict(context=ctx,error=str(exc),status=self.snapshot()))
                self.get_logger().error(str(exc))
                raise
        def _guided_place_for_auto(self):
            # This dataset ends at stable lift; no place or release motion.
            started=self.get_clock().now().nanoseconds/1e9
            deadline=time.monotonic()+20
            while self.get_clock().now().nanoseconds/1e9-started<2.2:
                if self._force_stop_event.is_set():raise RuntimeError('fragment stopped')
                if time.monotonic()>deadline:raise RuntimeError('hold simulation clock stalled')
                with self._lock:status=dict(self._status)
                if not valid_hold(status):raise RuntimeError('lift hold lost physical contact or stability')
                time.sleep(.05)
            return {'success':True,'message':'纠偏夹取抬升后保持2.2秒，短片段完成'}
        def start_recording(self):
            # The original auto workflow calls this before pregrasp motion.
            # Defer actual recording until the displaced state is reached.
            context()
            return {'success':True,'message':'录制将在偏离状态到达后开始'}
        def _execute_pregrasp_for_auto(self):
            response=super()._execute_pregrasp_for_auto()
            self._call_trigger(self._stop_motion_client,'/smolvla/stop_motion',timeout_s=5)
            Base.start_recording(self)
            time.sleep(.3)
            return response
    module.GraspGuiBridge=CorrectionGui;module.main()

def recorder_main():
    import z1_isaac_bridge.episode_recorder as module
    Base=module.EpisodeRecorder
    class CorrectionRecorder(Base):
        def __init__(self):
            self.correction_contact=False;super().__init__()
        def _task_status_callback(self,message):
            super()._task_status_callback(message)
            with self._lock:
                if self._recording:
                    self.correction_contact |= bool(self._task_status.get('left_finger_touching_table') or self._task_status.get('right_finger_touching_table'))
        def _start_callback(self,request,response):
            ctx=context();meta=json.loads(evidence_path(ctx).read_text())
            if meta['task_id']!=ctx['task']['id']:raise RuntimeError('纠正任务证据不匹配')
            self.correction_contact=False;self.require_task_success=False;response=super()._start_callback(request,response)
            if response.success:
                with self._lock:atomic_json(self._episode_dir/'correction.json',meta)
                self.correction_ctx=ctx
            return response
        def _save_callback(self,request,response):
            ctx=self.correction_ctx;meta=json.loads(evidence_path(ctx).read_text())
            with self._lock:
                if self.correction_contact or not meta.get('recovery_complete') or not valid_hold(self._task_status):
                    response.success=False;response.message='纠偏片段未通过无碰桌、恢复完成及物理抬升保持验收';return response
                meta.update(accepted=True,fragment_goal='correction_grasp_lift',split=ctx['task']['split'],final_status=dict(self._task_status),table_contact=False)
                atomic_json(self._episode_dir/'correction.json',meta)
            episode_index=int(self._episode_index)
            response=super()._save_callback(request,response)
            if response.success:
                root=self.dataset_root/f'episode_{episode_index:06d}'
                summary=json.loads((root/'episode_summary.json').read_text())
                summary.update(episode_goal='correction_grasp_lift',fragment_success=True,split=ctx['task']['split'],task_id=ctx['task']['id'])
                atomic_json(root/'episode_summary.json',summary)
                index_path=self.dataset_root/'episodes.jsonl'
                index_temp=index_path.with_suffix('.jsonl.tmp')
                with index_temp.open('w') as f:
                    for ep in sorted(self.dataset_root.glob('episode_[0-9]*')):
                        f.write(json.dumps(json.loads((ep/'episode_summary.json').read_text()),ensure_ascii=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                os.replace(index_temp,index_path)
            return response
    module.EpisodeRecorder=CorrectionRecorder;module.main()

if __name__=='__main__':
    role=sys.argv.pop(1)
    {'assistant':assistant_main,'gui':gui_main,'recorder':recorder_main}[role]()
