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
from protocol import atomic_json, check_scene, full_labels, tool_roll, validate_saved

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
            self.selecting_reference = False; self.approach_started = False

        def _command_scene_grasp(self, command, *args, **kwargs):
            if command == 'freeze' and self.approach_started:
                raise RuntimeError('Freezing target after approach is forbidden')
            return super()._command_scene_grasp(command, *args, **kwargs)

        def _generate_plan(self):
            self.cached = None; self.approach_started = False
            ctx = context()
            if ctx['phase'] == 'basic':
                return super()._generate_plan()
            self.prepared_shift = None
            if ctx['task']['kind'] == 'post_shift_realign':
                before = self._current_object_position('cup')
                self._command_scene_grasp('prepare_cup_shift', 'cup')
                if not self._wait_scene_grasp('CUP_SHIFT_READY', 'cup', 5):
                    raise RuntimeError('Prepared cup shift not confirmed')
                time.sleep(.4)
                after = self._current_object_position('cup')
                expected = ctx['original_correction']['actual_cup_position_m']
                if np.linalg.norm(np.array(after) - expected) > .002:
                    raise RuntimeError('Prepared cup shift differs from archived position')
                self.prepared_shift = dict(before=before, after=after,
                    actual_shift_world_m=(np.array(after)-before).tolist(), mode='setup_relocation_before_approach')
            self.selecting_reference = True
            try:
                message = super()._generate_plan()
            finally:
                self.selecting_reference = False
            from moveit_msgs.msg import RobotState
            routes = load_routes(ctx['source'])
            with self._lock:
                initial = [self._joint_positions[f'joint{i}'] for i in range(1,7)] + [self.gripper_open]
                approved = copy.deepcopy(self._approved_trajectory)
            end = routes.trajectory_points(approved.joint_trajectory, initial)[-1]
            p, r = self.geometry.link_fk(end); q = Rotation.from_matrix(r).as_quat().tolist()
            if abs(tool_roll(r) + 90.) > 3.:
                raise RuntimeError('Reference grasp plan is not minus90')
            state = RobotState(); state.is_diff = True
            state.joint_state.name = [f'joint{i}' for i in range(1,7)] + ['jointGripper']
            state.joint_state.position = initial
            deadline = time.monotonic() + 90
            def plan_route(route):
                current = copy.deepcopy(state); planned = []
                for step in route['preparation'] + route['recovery']:
                    self._raise_if_stop_requested()
                    if time.monotonic() > deadline:
                        raise RuntimeError('Recovery preplanning exceeded 90 seconds')
                    seed = dict(zip(current.joint_state.name, current.joint_state.position))
                    start = [seed[f'joint{i}'] for i in range(1,7)] + [self.gripper_open]
                    result = self._plan_candidate(dict(position=step['position'], orientation=step['orientation']),
                        linear=step['linear'], start_state=current, open_gripper=True,
                        position_tolerance=.002, orientation_tolerance=.025, velocity_scale=.10, acceleration_scale=.10)
                    self.geometry.check_path(routes.trajectory_points(result.planned_trajectory.joint_trajectory, start))
                    planned.append((step, result.planned_trajectory, start))
                    current = self._state_after_trajectory(result)
                return planned
            task = copy.deepcopy(ctx['task'])
            if self.prepared_shift:
                task['actual_shift_world_m'] = self.prepared_shift['actual_shift_world_m']
            route, planned, rejected = routes.choose_route(routes.routes(p, q, self._current_object_position('cup'), task), plan_route)
            self.cached = (route, planned, rejected); self.reference = (p.tolist(), q)
            return message + '; full correction route preplanned'

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
    module.GraspAssistant = RerecordAssistant
    module.main()


def gui_main():
    import z1_isaac_bridge.grasp_gui_bridge as module
    from std_msgs.msg import String
    module.GRID_LAYOUT_RESAMPLE_ATTEMPTS = 1
    Base = module.GraspGuiBridge
    class RerecordGui(Base):
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
