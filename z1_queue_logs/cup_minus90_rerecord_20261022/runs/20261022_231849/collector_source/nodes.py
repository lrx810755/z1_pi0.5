#!/usr/bin/python3
"""Full cup demonstrations; fixed archived layouts and minus90 grasp branch."""
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from protocol import atomic_json, check_scene, full_labels, tool_roll, validate_saved, preclose_correction, archived_shift_position

HERE = Path(__file__).resolve().parent
OLD = Path('/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_correction_fragments_20261019')
sys.path.append(str(OLD))
from geometry import Geometry


def context():
    return json.loads(Path(os.environ['Z1_RERECORD_CURRENT']).read_text())


def evidence_path(ctx):
    return Path(os.environ['Z1_RERECORD_CURRENT']).parent / ('evidence_' + ctx['attempt_id'] + '.json')


def load_routes(source):
    name = 'source_old_preflight' if source == 'old_correction' else 'source_new_preflight'
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def assistant_main():
    import z1_isaac_bridge.grasp_assistant as module
    module.GRASP_PROFILES = copy.deepcopy(module.GRASP_PROFILES)
    module.OBJECT_GRASP_PROFILES = copy.deepcopy(module.OBJECT_GRASP_PROFILES)
    module.GRASP_PROFILES['cup']['roll_deg'] = -90.
    module.GRASP_PROFILES['cup']['pregrasp_target_command'] = 'unfreeze'
    module.GRASP_PROFILES['cup']['single_pass_box_alignment'] = False
    # Use the same maximum planned excursion already used for loaded transport.
    # URDF limits and full path collision/velocity checks still apply.
    module.GRASP_PROFILES['cup']['max_joint_delta_rad'] = 3.0
    module.GRASP_PROFILES['cup']['accept_first_valid_candidate'] = False
    # Search additional approach directions at the same archived object poses.
    # Every candidate retains roll=-90 and passes the original collision checks.
    module.GRASP_PROFILES['cup']['extra_candidate_azimuths_deg'] = tuple(range(-180,180,15))
    module.GRASP_PROFILES['cup']['candidate_count'] = len(
        module.GRASP_PROFILES['cup']['candidate_azimuth_offsets_deg']) + 24
    module.OBJECT_GRASP_PROFILES['cup']['mirror_wrist_roll_candidates'] = False
    # Arrive within the release corridor during the collision-checked carry.
    # The original 55mm carry corridor needs a separate alignment to reach 20mm.
    module.BOX_TRANSPORT_XY_TOLERANCE_M = dict(module.BOX_TRANSPORT_XY_TOLERANCE_M, cup=.020)
    module.GRASP_CONFIG = copy.deepcopy(module.GRASP_CONFIG)
    module.GRASP_CONFIG['physical_test_lift']['goal_directed_max_attempts'] = 6
    Base = module.GraspAssistant
    class RerecordAssistant(Base):
        def __init__(self):
            super().__init__(); self.geometry = Geometry(); self.cached = None
            self.selecting_reference = False; self.approach_started = False; self.generating_grasp = False

        def _plan_candidate(self, candidate, linear=False, **kwargs):
            try:
                result = super()._plan_candidate(candidate, linear=linear, **kwargs)
            except RuntimeError as exc:
                if not (getattr(self,'generating_grasp',False) and linear
                        and module._is_recoverable_lift_planning_error(exc)):
                    raise
                self.get_logger().warning('Open-gripper final LIN blocked; plan a collision-checked free approach: '+str(exc))
                result = super()._plan_candidate(candidate, linear=False, **kwargs)
            if getattr(self,'generating_grasp',False) and candidate.get('final_grasp_target'):
                self.reference_candidates.append((copy.deepcopy(candidate),copy.deepcopy(result)))
            return result

        def _generate_grasp_reference(self):
            self.generating_grasp = True
            self.reference_candidates = []
            try:
                return super()._generate_plan()
            finally:
                self.generating_grasp = False

        def _command_scene_grasp(self, command, *args, **kwargs):
            if command == 'freeze' and self.approach_started:
                raise RuntimeError('Freezing target after approach is forbidden')
            return super()._command_scene_grasp(command, *args, **kwargs)

        def _generate_plan(self):
            self.cached = None; self.approach_started = False
            ctx = context()
            if ctx['phase'] == 'basic':
                return self._generate_grasp_reference()
            self.prepared_shift = None
            if ctx['task']['kind'] == 'post_shift_realign':
                before = self._current_object_position('cup')
                self._command_scene_grasp('prepare_cup_shift', 'cup')
                if not self._wait_scene_grasp('CUP_SHIFT_READY', 'cup', 5):
                    raise RuntimeError('Prepared cup shift not confirmed')
                time.sleep(.4)
                after = self._current_object_position('cup')
                expected = archived_shift_position(ctx['original_correction'])
                if np.linalg.norm(np.array(after) - expected) > .002:
                    raise RuntimeError('Prepared cup shift differs from archived position')
                self.prepared_shift = dict(before=before, after=after,
                    actual_shift_world_m=(np.array(after)-before).tolist(), mode='setup_relocation_before_approach')
            self.selecting_reference = True
            try:
                message = self._generate_grasp_reference()
            finally:
                self.selecting_reference = False
            from moveit_msgs.msg import RobotState
            from recovery_routes import side_entry_routes
            routes = load_routes(ctx['source'])
            with self._lock:
                initial = [self._joint_positions[f'joint{i}'] for i in range(1,7)] + [self.gripper_open]
                approved = copy.deepcopy(self._approved_trajectory)
                selected = copy.deepcopy(self._grasp_candidate)
            references=[(selected,approved)]
            for candidate,result in self.reference_candidates:
                if abs(candidate.get('azimuth_deg',0)-selected.get('azimuth_deg',0))>.1:
                    references.append((candidate,result.planned_trajectory))
            # On physical failure, the next round explores another valid
            # approach direction, preserving the scene and requested error.
            rotation=int(ctx.get('reference_trial',0))%len(references)
            references=references[rotation:]+references[:rotation]
            deadline=time.monotonic()+160
            reference_errors=[]
            state=RobotState();state.is_diff=True
            state.joint_state.name=[f'joint{i}' for i in range(1,7)]+['jointGripper']
            state.joint_state.position=initial
            def plan_route(route):
                current=copy.deepcopy(state);planned=[]
                for step in route['preparation']+route['recovery']:
                    self._raise_if_stop_requested()
                    if time.monotonic()>deadline:
                        raise RuntimeError('Recovery preplanning exceeded 160 seconds')
                    seed=dict(zip(current.joint_state.name,current.joint_state.position))
                    start=[seed[f'joint{i}'] for i in range(1,7)]+[self.gripper_open]
                    try:
                        result=self._plan_candidate(dict(position=step['position'],orientation=step['orientation']),
                            linear=step['linear'],start_state=current,open_gripper=True,
                            position_tolerance=.002,orientation_tolerance=.025,velocity_scale=.10,acceleration_scale=.10)
                        self.geometry.check_path(routes.trajectory_points(result.planned_trajectory.joint_trajectory,start))
                    except RuntimeError as exc:
                        detail=dict(stage=step['stage'],step=step,start=start,error=str(exc))
                        with evidence_path(ctx).with_suffix('.planning.jsonl').open('a') as stream:
                            stream.write(json.dumps(detail)+'\n')
                        raise RuntimeError(step['stage']+': '+str(exc)) from exc
                    planned.append((step,result.planned_trajectory,start))
                    current=self._state_after_trajectory(result)
                return planned
            task=copy.deepcopy(ctx['task'])
            if self.prepared_shift:
                task['actual_shift_world_m']=self.prepared_shift['actual_shift_world_m']
            for candidate,trajectory in references:
                if time.monotonic()>deadline:break
                end=routes.trajectory_points(trajectory.joint_trajectory,initial)[-1]
                p,r=self.geometry.link_fk(end);q=Rotation.from_matrix(r).as_quat().tolist()
                if abs(tool_roll(r)+90.)>3.:continue
                originals=list(routes.routes(p,q,self._current_object_position('cup'),task))
                alternatives=list(side_entry_routes(originals[0]))+originals
                try:
                    route,planned,rejected=routes.choose_route(alternatives,plan_route)
                except RuntimeError as exc:
                    reference_errors.append(dict(azimuth=candidate.get('azimuth_deg'),error=str(exc)))
                    continue
                self.cached=(route,planned,rejected);self.reference=(p.tolist(),q)
                with self._lock:self._grasp_candidate=copy.deepcopy(candidate)
                atomic_json(evidence_path(ctx).with_suffix('.reference.json'),dict(
                    selected=candidate,rejected_references=reference_errors,route=route))
                return message+'; full correction route preplanned'
            raise RuntimeError('No complete correction route among reference candidates: '+json.dumps(reference_errors,ensure_ascii=False))

        def _execute_plan(self):
            ctx = context()
            if ctx['phase'] == 'basic':
                # Production does its setup freeze before unfreezing and approaching.
                self.approach_started = False
                return super()._execute_plan()
            if self.cached is None:
                raise RuntimeError('Missing complete recovery preplan')
            route, planned, rejected = self.cached
            self._set_bottle_finger_collision(False); self._execute_gripper('open')
            self._command_scene_grasp('unfreeze', 'cup')
            if not self._wait_scene_grasp('UNFROZEN', '', 5):
                raise RuntimeError('Cup did not return to dynamic physics')
            before = self._current_object_position('cup')
            self.approach_started = True
            for step, trajectory, _ in planned[:len(route['preparation'])]:
                if self._send_arm_trajectory(trajectory):
                    raise RuntimeError('Correction preparation stopped')
            self._execute_gripper('open')
            p, q = self._current_ee_pose()
            if np.linalg.norm(np.array(self._current_object_position('cup')) - before) > .005:
                raise RuntimeError('Cup moved more than 5mm while preparing deviation')
            meta = dict(task_id=ctx['id'], task=ctx['task'], accepted=False, recovery_complete=False,
                reference_position=self.reference[0], reference_orientation=self.reference[1],
                requested_position=route['bad_position'], requested_orientation=route['bad_orientation'],
                actual_position=list(p), actual_orientation=list(q),
                preflight=dict(route=route, rejected_routes=rejected), stages=[],
                freeze_policy='pre_approach_only', prepared_cup_shift=self.prepared_shift)
            atomic_json(evidence_path(ctx), meta)
            with self._lock:
                self._approved_trajectory = None; self._approved_target = ''
            return 'Deviation prepared; continue recovery, grasp, transport and release'

        def _guided_grasp(self):
            ctx = context()
            if ctx['phase'] == 'correction':
                route, planned, _ = self.cached
                meta = json.loads(evidence_path(ctx).read_text())
                for step, trajectory, _ in planned[len(route['preparation']):]:
                    begin = self.get_clock().now().nanoseconds / 1e9
                    if self._send_arm_trajectory(trajectory):
                        raise RuntimeError('Correction recovery stopped')
                    meta['stages'].append(dict(stage=step['stage'], start_sim_time=begin,
                        end_sim_time=self.get_clock().now().nanoseconds / 1e9))
                meta['recovery_complete'] = True; atomic_json(evidence_path(ctx), meta)
            # Correct small measured controller/physics residuals with the cup
            # still dynamic. Do not widen the mandatory preclose envelope.
            profile = module.GRASP_PROFILES['cup']
            with self._lock:
                candidate = copy.deepcopy(self._grasp_candidate)
            desired = candidate.get('desired_object_in_stator_m',
                profile['calibrated_grasp_reference']['object_in_gripper_local_m'])
            adjustments = []
            for _ in range(2):
                self._command_scene_grasp('probe', 'cup')
                if not self._wait_scene_grasp('PROBED', 'cup', 2.):
                    raise RuntimeError('Missing live preclose position probe')
                with self._lock:
                    probe = dict(self._scene_grasp_status)
                actual = probe.get('local_position_m', [])
                correction = preclose_correction(actual, desired,
                    profile['single_alignment_tolerance_m'], profile['grasp_envelope_overrides'])
                if correction is None:
                    break
                p, q = self._current_ee_pose()
                world = module.rotate_vector(self._current_stator_orientation(), correction)
                goal = tuple(float(p[i]+world[i]) for i in range(3))
                self._execute_manual_pose(goal, q, '动态闭爪前毫米级残差校正', linear=True,
                    max_joint_delta=.15, position_tolerance=.001, orientation_tolerance=.01,
                    velocity_scale=.04, acceleration_scale=.04)
                adjustments.append(dict(measured_local_m=actual, correction_local_m=correction))
            atomic_json(evidence_path(ctx).with_suffix('.preclose.json'), dict(adjustments=adjustments))
            return super()._guided_grasp()

        def _carry_loaded_object_toward_box(self, target_object, loaded_envelope,
                minimum_object_z, stage, require_box_approach=False, preferred_object_z=None):
            # Leave 12mm of command headroom for object motion during wrist rotation;
            # measured height/contact acceptance remains the production requirement.
            preferred = max(minimum_object_z, self.box_approach_object_z,
                minimum_object_z if preferred_object_z is None else preferred_object_z) + .012
            return super()._carry_loaded_object_toward_box(target_object, loaded_envelope,
                minimum_object_z, stage, require_box_approach=require_box_approach,
                preferred_object_z=preferred)

        def _move_object_to(self, destination, description, linear,
                orientation_tolerance=None, max_joint_delta=None,
                velocity_scale=None, acceleration_scale=None):
            kwargs = dict(orientation_tolerance=orientation_tolerance,
                max_joint_delta=max_joint_delta, velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale)
            if not linear and '带载荷' in description:
                try:
                    return super()._move_object_to(destination, description, True, **kwargs)
                except RuntimeError as exc:
                    if not module._is_recoverable_lift_planning_error(exc):
                        raise
                    self.get_logger().warning('Loaded LIN plan unavailable; using original free planner: '+str(exc))
            return super()._move_object_to(destination, description, linear, **kwargs)
    module.GraspAssistant = RerecordAssistant
    module.main()


def gui_main():
    import z1_isaac_bridge.grasp_gui_bridge as module
    from std_msgs.msg import String
    module.GRID_LAYOUT_RESAMPLE_ATTEMPTS = 1
    Base = module.GraspGuiBridge
    class RerecordGui(Base):
        def _generate_pregrasp_for_auto(self):
            return self._call_trigger(self._generate_client,
                '/smolvla/auto_generate_pregrasp', timeout_s=240.)

        def randomize_episode(self, curriculum_level='full', target_grid=None):
            ctx = context()
            self._call_trigger(self._episode_reset_client, '/smolvla/reset_episode', timeout_s=60.)
            with self._lock:
                self._scene_randomization_status = {}
            command = String(); command.data = json.dumps(dict(seed=ctx['seed'], level='full',
                target_grid=ctx['grid'], settle_duration_s=2., non_target_max_angular_speed_rad_s=.35))
            self._randomization_command_pub.publish(command)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                with self._lock:
                    status = dict(self._scene_randomization_status)
                if status.get('phase') == 'ERROR':
                    raise ValueError('Archived scene reset failed: ' + str(status))
                if status.get('seed') == ctx['seed'] and status.get('phase') == 'RANDOMIZED':
                    proof = check_scene(ctx, status)
                    atomic_json(evidence_path(ctx).with_suffix('.scene.json'), dict(task_id=ctx['id'], verified=True, **proof))
                    return {'success': True, 'message': 'Archived scene reproduced and verified'}
                if self._force_stop_event.is_set():
                    raise RuntimeError('Stopped during archived scene reproduction')
                time.sleep(.05)
            raise TimeoutError('Archived scene reset timed out')

        def start_recording(self):
            if context()['phase'] == 'basic':
                return super().start_recording()
            return {'success': True, 'message': 'Recording starts at prepared deviation'}

        def _execute_pregrasp_for_auto(self):
            result = super()._execute_pregrasp_for_auto()
            if context()['phase'] == 'correction':
                Base.start_recording(self)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    with self._lock:
                        if self._recording_status.get('frame_count', 0) >= 3:
                            break
                    time.sleep(.05)
                else:
                    raise RuntimeError('Missing initial open-gripper observations')
            return result
        # Keep Base._guided_place_for_auto: full physical placement and release.
    module.GraspGuiBridge = RerecordGui
    module.main()


def recorder_main():
    import z1_isaac_bridge.episode_recorder as module
    Base = module.EpisodeRecorder
    class RerecordRecorder(Base):
        def __init__(self):
            self.contact = False; self.dynamic_violations = []; self.geometry = Geometry()
            super().__init__()

        def _task_status_callback(self, message):
            super()._task_status_callback(message)
            with self._lock:
                if self._recording:
                    self.contact |= bool(self._task_status.get('left_finger_touching_table') or self._task_status.get('right_finger_touching_table'))
                    if context()['phase'] == 'correction' and (self._task_status.get('frozen') or not self._task_status.get('target_collisions_enabled')):
                        self.dynamic_violations.append(self.get_clock().now().nanoseconds / 1e9)

        def _start_callback(self, req, res):
            self.contact = False; self.dynamic_violations = []; self.require_task_success = True
            self.ctx = context()
            res = super()._start_callback(req, res)
            if res.success and self.ctx['phase'] == 'correction':
                meta = json.loads(evidence_path(self.ctx).read_text())
                if meta.get('prepared_cup_shift'):
                    with self._lock:
                        self._episode_randomization = copy.deepcopy(self._episode_randomization)
                        self._episode_randomization['original_object_positions_m'] = copy.deepcopy(
                            self._episode_randomization['object_positions_m'])
                        self._episode_randomization['object_positions_m']['cup'] = meta['prepared_cup_shift']['after']
                        self._episode_randomization['prepared_cup_shift'] = meta['prepared_cup_shift']
                        path = self._episode_dir / 'episode_start.json'
                        start = json.loads(path.read_text())
                        start['scene_randomization'] = self._episode_randomization
                        atomic_json(path, start)
            return res

        def _save_callback(self, req, res):
            ctx = self.ctx
            with self._lock:
                self._frames_file.flush()
                rows = [json.loads(l) for l in (self._episode_dir / 'frames.jsonl').read_text().splitlines()]
                try:
                    labels, close, release = full_labels(rows)
                    _, rotation = self.geometry.link_fk(rows[max(0, close-1)]['observation.state'][:6])
                    roll = tool_roll(rotation)
                    validate_saved(self._task_status, roll, self.contact, len(rows)-release)
                    if self.dynamic_violations:
                        raise ValueError('Target frozen or collisions disabled during correction')
                    proof = json.loads(evidence_path(ctx).with_suffix('.scene.json').read_text())
                    if not proof['verified'] or proof['task_id'] != ctx['id']:
                        raise ValueError('Missing archived-scene verification')
                    meta = json.loads(evidence_path(ctx).read_text()) if ctx['phase'] == 'correction' else {}
                    if ctx['phase'] == 'correction' and not meta.get('recovery_complete'):
                        raise ValueError('Recovery not complete')
                except (ValueError, KeyError, OSError) as exc:
                    res.success = False; res.message = 'Rerecord acceptance rejected: ' + str(exc)
                    atomic_json(evidence_path(ctx).with_suffix('.rejection.json'), dict(error=str(exc), status=self._task_status))
                    return res
                meta.update(accepted=True, original_path=ctx['original_path'], source=ctx['source'], task_id=ctx['id'],
                    split=ctx['split'], plan_hash=ctx['plan_hash'], phase=ctx['phase'], attempt_id=ctx['attempt_id'],
                    episode_goal='physical_pick_place', close_command_frame=close, open_command_frame=release,
                    measured_preclose_roll_deg=roll, scene_reproduction=proof, table_contact=False,
                    collector_manifest=ctx.get('collector_manifest'),
                    final_status=dict(self._task_status), dynamic_violations=self.dynamic_violations)
                atomic_json(self._episode_dir / 'rerecord.json', meta)
                if ctx['phase'] == 'correction':
                    atomic_json(self._episode_dir / 'correction.json', meta)
                atomic_json(self._episode_dir / 'gripper_labels.json', dict(labels=labels,
                    close_command_frame=close, open_command_frame=release, raw_state_encoding='actual_joint_angles'))
                return super()._save_callback(req, res)
    module.EpisodeRecorder = RerecordRecorder
    module.main()


if __name__ == '__main__':
    role = sys.argv.pop(1)
    {'assistant': assistant_main, 'gui': gui_main, 'recorder': recorder_main}[role]()
