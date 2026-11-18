#!/usr/bin/python3
"""Task-local full-recovery nodes; production source is never patched."""
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from protocol import (atomic_json, command_labels, training_mask,
                      validate_category, validate_final, validate_preparation, validate_withdrawal,
                      wait_simulation_duration)
from recovery_variants import elevated_approach_routes

LEGACY = Path('/home/lp531/z1_queue_logs/cup_minus90_rerecord_20261022/collector')
sys.path.append('/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_correction_fragments_20261019')
from geometry import Geometry


def load_helper(name):
    spec = importlib.util.spec_from_file_location('correction100_' + name, LEGACY / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROUTES = load_helper('source_new_preflight')
SIDE_ROUTES = load_helper('recovery_routes')
LEGACY_PROTOCOL = load_helper('protocol')


def context():
    return json.loads(Path(os.environ['Z1_CORRECTION100_CURRENT']).read_text())


def evidence_path(ctx):
    return Path(os.environ['Z1_CORRECTION100_CURRENT']).parent / ('evidence_' + ctx['attempt_id'] + '.json')


def assistant_main():
    import z1_isaac_bridge.grasp_assistant as module
    module.GRASP_PROFILES = copy.deepcopy(module.GRASP_PROFILES)
    module.OBJECT_GRASP_PROFILES = copy.deepcopy(module.OBJECT_GRASP_PROFILES)
    module.GRASP_PROFILES['cup'].update(
        roll_deg=-90., pregrasp_target_command='unfreeze', single_pass_box_alignment=False,
        max_joint_delta_rad=3.0)
    module.OBJECT_GRASP_PROFILES['cup'].update(
        mirror_wrist_roll_candidates=False, loaded_transport_velocity_scale=.05,
        loaded_transport_acceleration_scale=.02)
    module.BOX_TRANSPORT_XY_TOLERANCE_M = dict(module.BOX_TRANSPORT_XY_TOLERANCE_M, cup=.020)
    module.GRASP_CONFIG = copy.deepcopy(module.GRASP_CONFIG)
    module.GRASP_CONFIG['physical_test_lift']['goal_directed_max_attempts'] = 10
    Base = module.GraspAssistant

    class CorrectionAssistant(Base):
        def __init__(self):
            self.approach_started = False
            self.preparation_contact_seen = False
            self.meta = None
            super().__init__()
            self.geometry = Geometry()
            self.meta = None
            self.prepared = None
            self.approach_started = False

        def _task_status_callback(self, message):
            super()._task_status_callback(message)
            with self._lock:
                if self.approach_started:
                    self.preparation_contact_seen |= bool(
                        self._task_status.get('left_finger_touching_table')
                        or self._task_status.get('right_finger_touching_table'))

        def _execute_manual_pose(self, position, orientation, description, *args, **kwargs):
            retreat = description == '释放后垂直撤离'
            if not retreat:
                return super()._execute_manual_pose(position, orientation, description, *args, **kwargs)
            before = self._current_ee_pose()[0]
            try:
                # Same final4cm withdrawal and original per-motion joint limit;
                # three smaller steps avoid rejecting a valid0.485rad whole move.
                for fraction in (1 / 3, 2 / 3, 1.):
                    waypoint = tuple(float(before[i] + fraction * (position[i] - before[i])) for i in range(3))
                    result = super()._execute_manual_pose(waypoint, orientation,
                        description + '分段', *args, **kwargs)
                return result
            finally:
                after = self._current_ee_pose()[0]
                self.meta['withdrawal_rise_m'] = float(after[2] - before[2])
                self.event('withdrawal_measured', before=list(before), after=list(after))

        def event(self, event, **values):
            if self.meta is not None:
                self.meta['events'].append(dict(event=event,
                    time=self.get_clock().now().nanoseconds * 1e-9, **values))
                atomic_json(evidence_path(self.ctx), self.meta)

        def _command_scene_grasp(self, command, *args, **kwargs):
            if command in ('freeze', 'prepare_cup_shift') and self.approach_started:
                raise RuntimeError('Freeze/relocation forbidden after approach begins')
            return super()._command_scene_grasp(command, *args, **kwargs)

        def _execute_gripper_target(self, state, target):
            self.event('gripper_intent', state=state, target=float(target))
            return super()._execute_gripper_target(state, target)

        def _execute_gripper_close_until_obstructed(self, target, target_object, **kwargs):
            self.event('gripper_intent', state='close', target=float(target))
            return super()._execute_gripper_close_until_obstructed(target, target_object, **kwargs)

        def _generate_plan(self):
            from moveit_msgs.msg import RobotState
            self.ctx = context()
            task = self.ctx['task']
            self.approach_started = False
            self.preparation_contact_seen = False
            self.prepared = None
            self.meta = dict(task_id=task['id'], task=task, source_type=task['source_type'],
                accepted=False, recovery_complete=False, events=[], recovery_stages=[])
            if task['kind'] == 'shifted_target':
                before = self._current_object_position('cup')
                self._command_scene_grasp('prepare_cup_shift', 'cup')
                if not self._wait_scene_grasp('CUP_SHIFT_READY', 'cup', 5.):
                    raise RuntimeError('Target setup relocation was not acknowledged')
                time.sleep(.4)
                after = self._current_object_position('cup')
                shift = (np.asarray(after) - before).tolist()
                self.meta['prepared_cup_shift'] = dict(before=list(before), after=list(after),
                    actual_shift_world_m=shift, distance_m=float(np.linalg.norm(shift)),
                    mode='setup_relocation_before_approach_and_recording')
            message = super()._generate_plan()
            with self._lock:
                initial = [self._joint_positions[f'joint{i}'] for i in range(1, 7)] + [self.gripper_open]
                approved = copy.deepcopy(self._approved_trajectory)
                self.reference_candidate = copy.deepcopy(self._grasp_candidate)
            end = ROUTES.trajectory_points(approved.joint_trajectory, initial)[-1]
            position, rotation = self.geometry.link_fk(end)
            orientation = Rotation.from_matrix(rotation).as_quat().tolist()
            if abs(LEGACY_PROTOCOL.tool_roll(rotation) + 90.) > 3.:
                raise RuntimeError('Reference FK is not the minus90 branch')
            self.reference = (position, orientation)
            route_task = copy.deepcopy(task)
            route_task['kind'] = task['route_kind']
            if task['kind'] == 'shifted_target':
                route_task['kind'] = 'post_shift_realign'
                route_task['actual_shift_world_m'] = self.meta['prepared_cup_shift']['actual_shift_world_m']
            state = RobotState()
            state.is_diff = True
            state.joint_state.name = [f'joint{i}' for i in range(1, 7)] + ['jointGripper']
            state.joint_state.position = initial
            deadline = time.monotonic() + 160

            def plan_route(route):
                current, planned = copy.deepcopy(state), []
                for step in route['preparation'] + route['recovery']:
                    self._raise_if_stop_requested()
                    if time.monotonic() > deadline:
                        raise RuntimeError('Correction route preplanning exceeded160 seconds')
                    seed = dict(zip(current.joint_state.name, current.joint_state.position))
                    start = [seed[f'joint{i}'] for i in range(1, 7)] + [self.gripper_open]
                    try:
                        result = self._plan_candidate(dict(position=step['position'], orientation=step['orientation']),
                            linear=step['linear'], start_state=current, open_gripper=True,
                            position_tolerance=.002, orientation_tolerance=.025,
                            velocity_scale=.10, acceleration_scale=.10)
                        self.geometry.check_path(ROUTES.trajectory_points(result.planned_trajectory.joint_trajectory, start))
                    except RuntimeError as exc:
                        detail = f"{route['strategy']}:{step['stage']} position={step['position']}: {exc}"
                        self.get_logger().warning('CORRECTION_ROUTE_REJECTED ' + detail)
                        raise RuntimeError(detail) from exc
                    planned.append((step, result.planned_trajectory))
                    current = self._state_after_trajectory(result)
                return planned

            if task['kind'] in ('close_transition', 'transport', 'placement'):
                # These categories need a normal preclose state; their correction
                # is the close decision or a measured later loaded-state deviation.
                # Reuse the collision-checked approved approach rather than asking
                # a different OMPL path to the same endpoint and sweeping the cup.
                step = dict(stage='prepare_reference', position=position.tolist(),
                            orientation=orientation, linear=False)
                route = dict(bad_position=position.tolist(), bad_orientation=orientation,
                             preparation=[step], recovery=[], strategy='approved_reference')
                planned, rejected = [(step, approved)], []
                self.meta['pregrasp_mode'] = 'approved_reference_no_initial_offset'
            else:
                direct = list(ROUTES.routes(position, orientation, self._current_object_position('cup'), route_task))
                candidates = list(SIDE_ROUTES.side_entry_routes(direct[0])) + direct
                if task['kind'] == 'approach':
                    # Requested same-height lateral errors can put a finger inside
                    # the cup's solid collision body. Keep their XY error and
                    # prepare at a logged, safe height, then align before descent.
                    candidates = list(elevated_approach_routes(direct[0], position, orientation)) + candidates
                sys.path.insert(0,'/home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031')
                from schedule import prioritize_routes
                preferences=json.loads(Path('/home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031/preferred_route_strategies.json').read_text())
                candidates=prioritize_routes(candidates,preferences.get(task['id']))
                route, planned, rejected = ROUTES.choose_route(candidates, plan_route)
            self.prepared = (route, planned)
            self.meta.update(reference_position_m=position.tolist(), reference_orientation_xyzw=orientation,
                requested_start_position_m=route['bad_position'], preflight=dict(route=route, rejected=rejected))
            atomic_json(evidence_path(self.ctx), self.meta)
            return message + '; complete correction route preplanned'

        def _execute_plan(self):
            if self.prepared is None:
                raise RuntimeError('Missing preparation and recovery plan')
            route, planned = self.prepared
            self._set_bottle_finger_collision(False)
            self._execute_gripper('open')
            self._command_scene_grasp('unfreeze', 'cup')
            if not self._wait_scene_grasp('UNFROZEN', '', 5.):
                raise RuntimeError('Cup is not dynamically simulated')
            self.approach_started = True
            before = self._current_object_position('cup')
            for step, trajectory in planned[:len(route['preparation'])]:
                if self._send_arm_trajectory(trajectory):
                    raise RuntimeError('Correction preparation stopped')
            self._execute_gripper('open')
            if np.linalg.norm(np.asarray(self._current_object_position('cup')) - before) > .005:
                raise RuntimeError('Preparation unintentionally displaced cup by more than5mm')
            p, q = self._current_ee_pose()
            delta = np.asarray(p) - self.reference[0]
            with self._lock:
                joints = [self._joint_positions[f'joint{i}'] for i in range(1, 7)] + [self._joint_positions['jointGripper']]
            self.meta.update(actual_start_position_m=list(p), actual_start_orientation_xyzw=list(q),
                measured_start_deviation_m=float(np.linalg.norm(delta)),
                measured_world_z_deviation_m=float(delta[2]),
                measured_finger_table_clearance_m=self.geometry.height(joints),
                actual_cup_position_at_recording_m=list(self._current_object_position('cup')))
            self.meta['preparation_table_contact_seen'] = self.preparation_contact_seen
            validate_preparation(self.meta['measured_finger_table_clearance_m'], self.preparation_contact_seen)
            atomic_json(evidence_path(self.ctx), self.meta)
            with self._lock:
                self._approved_trajectory = None
                self._approved_target = ''
            return 'Prepared deviation; record continuous recovery to box release'

        def _guided_grasp(self):
            route, planned = self.prepared
            if self.ctx['task']['kind'] == 'regrasp':
                # Create and measure a single-finger clamp at the offset pose, then
                # supervise reopening and the already-preplanned return to alignment.
                self.event('intervention_start', purpose='prepare_unilateral_partial_clamp')
                unilateral = False
                for target in np.arange(-.86, -.49, .02):
                    # These are bounded angle steps: reaching the target without
                    # obstruction is valid and is followed by a fresh contact probe.
                    self._execute_gripper('close', target_override=float(target))
                    time.sleep(.1)
                    self._command_scene_grasp('probe', 'cup')
                    if not self._wait_scene_grasp('PROBED', 'cup', 3.):
                        raise RuntimeError('Missing retry contact probe')
                    with self._lock:
                        probe = dict(self._scene_grasp_status)
                    left = probe.get('left_finger_contact_object') == 'cup'
                    right = probe.get('right_finger_contact_object') == 'cup'
                    if left != right:
                        unilateral = True
                        break
                    if left and right:
                        raise RuntimeError('Healthy bilateral clamp cannot count as unilateral regrasp')
                self.meta.update(retry_contact_seen=unilateral, retry_unilateral_contact=unilateral,
                                 retry_probe=probe)
                self.event('intervention_end')
                if not unilateral:
                    raise RuntimeError('No measured unilateral contact for regrasp')
                self._execute_gripper('open')
                self.meta['retry_performed'] = True
                self.event('retry_opened')
            for step, trajectory in planned[len(route['preparation']):]:
                begin = self.get_clock().now().nanoseconds * 1e-9
                if self._send_arm_trajectory(trajectory):
                    raise RuntimeError('Recovery motion stopped')
                self.meta['recovery_stages'].append(dict(stage=step['stage'], start=begin,
                    end=self.get_clock().now().nanoseconds * 1e-9,
                    ee_position_m=list(self._current_ee_pose()[0])))
            self.meta['recovery_complete'] = True
            self.meta['relocalized'] = self.ctx['task']['kind'] == 'shifted_target'
            atomic_json(evidence_path(self.ctx), self.meta)
            self._correct_preclose_residual()
            result = super()._guided_grasp()
            self.meta['close_transition_verified'] = True
            atomic_json(evidence_path(self.ctx), self.meta)
            return result

        def _correct_preclose_residual(self):
            profile = module.GRASP_PROFILES['cup']
            desired = self.reference_candidate.get('desired_object_in_stator_m',
                profile['calibrated_grasp_reference']['object_in_gripper_local_m'])
            for _ in range(2):
                self._command_scene_grasp('probe', 'cup')
                if not self._wait_scene_grasp('PROBED', 'cup', 2.):
                    raise RuntimeError('Missing preclose residual probe')
                with self._lock:
                    actual = self._scene_grasp_status.get('local_position_m', [])
                correction = LEGACY_PROTOCOL.preclose_correction(actual, desired,
                    profile['single_alignment_tolerance_m'], profile['grasp_envelope_overrides'])
                if correction is None:
                    return
                p, q = self._current_ee_pose()
                delta = module.rotate_vector(self._current_stator_orientation(), correction)
                goal = tuple(float(p[i] + delta[i]) for i in range(3))
                self._execute_manual_pose(goal, q, '动态闭爪前残差校正', linear=True,
                    max_joint_delta=.15, position_tolerance=.001, orientation_tolerance=.01,
                    velocity_scale=.04, acceleration_scale=.04)

        def _guided_place(self):
            kind = self.ctx['task']['kind']
            if kind in ('transport', 'placement'):
                envelope = dict(module.GRASP_ENVELOPE)
                envelope.update(module.GRASP_PROFILES['cup'].get('grasp_envelope_overrides', {}))
                self._require_loaded_physical_grasp('cup', envelope, '纠偏扰动前夹持确认')
                before = self._current_object_position('cup')
                delta = self.ctx['task']['loaded_offset_world_m']
                destination = tuple(float(before[i] + delta[i]) for i in range(3))
                if destination[2] > .215:
                    raise RuntimeError('Loaded correction setup exceeds215mm object height')
                self.event('intervention_start', purpose=kind + '_state_preparation')
                for fraction in (.5, 1.):
                    waypoint = tuple(float(before[i] + fraction * delta[i]) for i in range(3))
                    try:
                        self._move_object_to(waypoint, '纠偏状态准备', linear=True,
                            orientation_tolerance=.08, max_joint_delta=.5,
                            velocity_scale=.04, acceleration_scale=.04)
                    except RuntimeError as exc:
                        if not module._is_recoverable_lift_planning_error(exc):
                            raise
                        self._move_object_to(waypoint, '纠偏状态准备备用路径', linear=False,
                            orientation_tolerance=.08, max_joint_delta=.5,
                            velocity_scale=.04, acceleration_scale=.04)
                    self._require_loaded_physical_grasp('cup', envelope, '纠偏状态准备分段后')
                self._require_loaded_physical_grasp('cup', envelope, '纠偏扰动后夹持确认')
                after = self._current_object_position('cup')
                self.meta.update(loaded_before_m=list(before), loaded_after_m=list(after),
                    measured_disturbance_m=float(np.linalg.norm(np.asarray(after) - before)))
                if kind == 'placement':
                    self.meta.update(placement_vertical_excess_m=float(after[2] - before[2]),
                        placement_xy_error_m=float(np.linalg.norm(np.asarray(after[:2]) - self.box_center[:2])))
                    self.meta['placement_hover_duration_s'] = wait_simulation_duration(
                        lambda: self.get_clock().now().nanoseconds * 1e-9, .4,
                        check_stop=self._raise_if_stop_requested)
                self.event('intervention_end')
                self.event('loaded_recovery_start', kind=kind)
            result = super()._guided_place()
            if kind in ('transport', 'placement'):
                self.meta['recovery_after_disturbance'] = True
                self.event('loaded_recovery_complete', kind=kind)
            return result

        def _carry_loaded_object_toward_box(self, target_object, loaded_envelope,
                minimum_object_z, stage, require_box_approach=False, preferred_object_z=None):
            preferred = max(minimum_object_z, self.box_approach_object_z,
                minimum_object_z if preferred_object_z is None else preferred_object_z) + .012
            return super()._carry_loaded_object_toward_box(target_object, loaded_envelope,
                minimum_object_z, stage, require_box_approach=require_box_approach,
                preferred_object_z=preferred)

    module.GraspAssistant = CorrectionAssistant
    module.main()


def gui_main():
    import z1_isaac_bridge.grasp_gui_bridge as module
    from std_msgs.msg import String
    Base = module.GraspGuiBridge
    module.GRID_LAYOUT_RESAMPLE_ATTEMPTS = 2

    class CorrectionGui(Base):
        def randomize_episode(self, curriculum_level='full', target_grid=None):
            ctx = context()
            task = ctx['task']
            # Retry other already-successful basic layouts within the same cell;
            # record the selected seed/path before assistant planning reads context.
            reference_root = Path('/home/lp531/smolvla_data/z1_tabletop_raw_cup_basic_4x5x20_renderfix_20261026')
            choices = []
            for folder in sorted(reference_root.glob('episode_[0-9]*')):
                summary = json.loads((folder / 'episode_summary.json').read_text())
                if summary.get('task_success') and summary['scene_randomization']['target_grid'] == task['grid']:
                    choices.append((folder, summary))
            initial = task.setdefault('reference_initial_episode', task['reference_basic_episode'])
            base_index = next(i for i, (folder, _) in enumerate(choices) if str(folder) == initial)
            trial = ctx.get('layout_trial', 0)
            sys.path.insert(0,'/home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031')
            from schedule import reference_index
            preferred_map=json.loads(Path('/home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031/preferred_reference_layouts.json').read_text())
            preferred_path=preferred_map.get(task['id'])
            preferred_index=next((i for i,(p,_) in enumerate(choices) if str(p)==preferred_path),None)
            folder, summary = choices[reference_index(base_index,ctx['round'],trial,len(choices),preferred_index)]
            ctx['layout_trial'] = trial + 1
            task.update(seed=summary['scene_randomization']['seed'], reference_basic_episode=str(folder),
                        reference_scene=summary['scene_randomization'])
            atomic_json(Path(os.environ['Z1_CORRECTION100_CURRENT']), ctx)
            self._call_trigger(self._episode_reset_client, '/smolvla/reset_episode', timeout_s=60.)
            with self._lock:
                self._scene_randomization_status = {}
            command = String()
            command.data = json.dumps(dict(seed=task['seed'], level='full', target_grid=task['grid'],
                settle_duration_s=2., non_target_max_angular_speed_rad_s=.35))
            self._randomization_command_pub.publish(command)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                with self._lock:
                    state = dict(self._scene_randomization_status)
                if state.get('phase') == 'ERROR':
                    raise RuntimeError('Reference scene reset failed: ' + str(state))
                if state.get('phase') == 'RANDOMIZED' and state.get('seed') == task['seed']:
                    if state.get('target_grid') != task['grid']:
                        raise RuntimeError('Reference grid mismatch')
                    return {'success': True, 'message': 'Successful basic-layout seed restored for new correction'}
                if self._force_stop_event.is_set():
                    raise RuntimeError('Stopped during reference-layout reset')
                time.sleep(.05)
            raise RuntimeError('Reference-layout reset timed out')

        def _generate_pregrasp_for_auto(self):
            return self._call_trigger(self._generate_client,
                '/smolvla/auto_generate_pregrasp', timeout_s=180.)

        def start_recording(self):
            return {'success': True, 'message': 'Recording starts at measured deviation'}

        def _execute_pregrasp_for_auto(self):
            result = super()._execute_pregrasp_for_auto()
            Base.start_recording(self)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                with self._lock:
                    if self._recording_status.get('frame_count', 0) >= 6:
                        return result
                time.sleep(.05)
            raise RuntimeError('Missing three initial open-gripper frames')

        def _guided_place_for_auto(self):
            result = super()._guided_place_for_auto()
            self._call_trigger(self._stop_motion_client, '/smolvla/stop_motion', timeout_s=5.)
            wait_simulation_duration(lambda: self.get_clock().now().nanoseconds * 1e-9, 2.2)
            return result

    module.GraspGuiBridge = CorrectionGui
    module.main()


def recorder_main():
    import z1_isaac_bridge.episode_recorder as module
    sys.path.insert(0,'/home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031')
    from common import guarded_recorder
    Base = guarded_recorder(module.EpisodeRecorder,"correction")

    class CorrectionRecorder(Base):
        def __init__(self):
            self.geometry = Geometry()
            self.contact = False
            self.dynamic_samples = 0
            self.dynamic_violations = []
            self.contact_stream = None
            super().__init__()

        def _task_status_callback(self, message):
            super()._task_status_callback(message)
            with self._lock:
                if self._recording:
                    status = dict(self._task_status)
                    self.dynamic_samples += 1
                    self.contact |= bool(status.get('left_finger_touching_table') or status.get('right_finger_touching_table'))
                    if status.get('frozen') is not False or status.get('target_collisions_enabled') is not True:
                        if len(self.dynamic_violations) < 20:
                            self.dynamic_violations.append(status)
                    if self.contact_stream is not None:
                        self.contact_stream.write(json.dumps(dict(
                            simulation_time=self.get_clock().now().nanoseconds * 1e-9,
                            status=status), ensure_ascii=False) + '\n')

        def _start_callback(self, req, res):
            with self._lock:
                self.ctx = context()
                meta = json.loads(evidence_path(self.ctx).read_text())
                self.contact, self.dynamic_samples, self.dynamic_violations = False, 0, []
                self.require_task_success = True
                res = super()._start_callback(req, res)
                if res.success:
                    with self._lock:
                        if self.contact_stream is not None:
                            self.contact_stream.close()
                        self.contact_stream = (self._episode_dir / 'physics_status.jsonl').open('w')
                        shift = meta.get('prepared_cup_shift')
                        if shift:
                            self._episode_randomization = copy.deepcopy(self._episode_randomization)
                            self._episode_randomization['original_object_positions_m'] = copy.deepcopy(
                                self._episode_randomization['object_positions_m'])
                            self._episode_randomization['object_positions_m']['cup'] = shift['after']
                            self._episode_randomization['prepared_cup_shift'] = shift
                            start = json.loads((self._episode_dir / 'episode_start.json').read_text())
                            start['scene_randomization'] = self._episode_randomization
                            atomic_json(self._episode_dir / 'episode_start.json', start)
                return res

        def _discard_callback(self, req, res):
            with self._lock:
                if self.contact_stream is not None:
                    self.contact_stream.close()
                    self.contact_stream = None
            return super()._discard_callback(req, res)

        def _save_callback(self, req, res):
            with self._lock:
                try:
                    self._frames_file.flush()
                    rows = [json.loads(line) for line in (self._episode_dir / 'frames.jsonl').read_text().splitlines()]
                    meta = json.loads(evidence_path(self.ctx).read_text())
                    labels, close, release = command_labels(rows, meta['events'])
                    _, rotation = self.geometry.link_fk(rows[max(0, close - 1)]['observation.state'][:6])
                    roll = LEGACY_PROTOCOL.tool_roll(rotation)
                    validate_final(self._task_status, roll, self.contact,
                        self.dynamic_samples, self.dynamic_violations, len(rows) - release)
                    validate_category(self.ctx['task'], meta)
                    validate_preparation(meta['measured_finger_table_clearance_m'],
                                         meta.get('preparation_table_contact_seen', True))
                    validate_withdrawal(meta)
                    mask, anchors = training_mask(rows, meta['events'])
                    if not any(anchors):
                        raise ValueError('No eligible16-step training window')
                except (OSError, ValueError, KeyError) as exc:
                    res.success, res.message = False, 'Correction100 acceptance rejected: ' + str(exc)
                    atomic_json(evidence_path(self.ctx).with_suffix('.rejection.json'),
                        dict(error=str(exc), task=self.ctx['task'], status=self._task_status))
                    return res
                meta.update(accepted=True, task_id=self.ctx['task']['id'], split=self.ctx['task']['split'],
                    plan_hash=self.ctx['plan_hash'], attempt_id=self.ctx['attempt_id'],
                    measured_preclose_roll_deg=roll, close_command_frame=close,
                    open_command_frame=release, dynamic_samples=self.dynamic_samples,
                    dynamic_violations=self.dynamic_violations, table_contact=False,
                    final_status=dict(self._task_status), eligible_action_horizon16_anchors=sum(anchors))
                atomic_json(self._episode_dir / 'correction.json', meta)
                atomic_json(self._episode_dir / 'gripper_labels.json', dict(labels=labels,
                    close_command_frame=close, open_command_frame=release,
                    encoding='command-time intent; raw actual state/actions retained'))
                atomic_json(self._episode_dir / 'training_mask.json', dict(
                    expert_frame_mask=mask, valid_action_horizon16_anchors=anchors,
                    instruction='Exclude deliberate intervention actions and every action window intersecting them.'))
                if self.contact_stream is not None:
                    self.contact_stream.flush()
                    self.contact_stream.close()
                    self.contact_stream = None
                return super()._save_callback(req, res)

    module.EpisodeRecorder = CorrectionRecorder
    module.main()


if __name__ == '__main__':
    role = sys.argv.pop(1)
    {'assistant': assistant_main, 'gui': gui_main, 'recorder': recorder_main}[role]()
