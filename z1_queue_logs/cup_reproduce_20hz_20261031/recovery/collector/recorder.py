#!/usr/bin/python3
"""Private recorder for expert recovery from actual policy-pushed cup states.

ROS and URDF geometry are loaded only by main()/make_recorder_class(); the
acceptance boundary is independently testable without a running scene.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
OLD_COLLECTOR = Path('/home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031/correction/collector')
OLD_MINUS90 = Path('/home/lp531/z1_queue_logs/cup_minus90_rerecord_20261022/collector')
GEOMETRY_PATH = Path('/home/lp531/ros2_ws/src/z1_ros2/scripts/cup_correction_fragments_20261019/geometry.py')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load('push_recovery_old_protocol', OLD_COLLECTOR/'protocol.py')


def context():
    return json.loads(Path(os.environ.get('Z1_PUSH_RECOVERY_CURRENT', ROOT/'current.json')).read_text())


def evidence_path(ctx):
    return Path(os.environ.get('Z1_PUSH_RECOVERY_CURRENT', ROOT/'current.json')).parent / ('evidence_'+ctx['attempt_id']+'.json')


def source_scene(ctx):
    scene = copy.deepcopy(ctx['candidate']['raw_initial_scene_randomization'])
    scene['original_target_grid'] = copy.deepcopy(scene.get('target_grid'))
    scene['source_grid'] = copy.deepcopy(ctx['task']['grid'])
    scene['target_grid'] = copy.deepcopy(ctx['task']['grid'])
    # Cup remains at its ORIGINAL randomized position in this scene metadata;
    # the actual post-policy takeover cup pose is separately stored in correction.
    return scene


def physics_violation(status):
    if status.get('frozen') is not False or status.get('target_collisions_enabled') is not True:
        return 'non_dynamic_or_disabled_collision'
    if status.get('left_finger_touching_table') or status.get('right_finger_touching_table'):
        return 'finger_table_contact'
    return None


def stable_release(status):
    return (status.get('inside_success_volume') is True
        and status.get('released') is True
        and status.get('holding_observed') is False
        and status.get('gripper_commanded_closed') is False
        and float(status.get('gripper_position_rad',math.inf)) <= -.88
        and float(status.get('object_speed_m_s',math.inf)) <= .02
        and physics_violation(status) is None)


def validate_start(ctx, meta, status):
    if not (meta.get('recording_ready') is True or meta.get('expert_hold_verified') is True):
        raise ValueError('expert hold verification required before recording')
    if ctx['task']['kind'] != 'push_recovery':
        raise ValueError('unexpected recovery task kind')
    if ctx['candidate'].get('source_final_box_success') is not False:
        raise ValueError('source rollout must be an authoritative final-box failure')
    if physics_violation(status):
        raise ValueError('expert hold is not dynamic and free of finger/table contact')
    if (status.get('gripper_commanded_closed') or status.get('physical_grasp_ok')
            or status.get('holding_observed') or status.get('was_grasped')):
        raise ValueError('expert hold must precede grasp with open gripper')
    if (status.get('cylinder_tilt_valid') is not True
            or float(status.get('cylinder_tilt_deg', math.inf)) > 10.
            or status.get('object_touching_table') is not True
            or float(status.get('object_speed_m_s', math.inf)) > .02):
        raise ValueError('expert hold cup must be upright, on table and settled')


def _time(meta, keys, label):
    for key in keys:
        value = meta.get(key)
        if isinstance(value, (float,int)) and math.isfinite(value):
            return float(value)
    # Compatibility with named stages in the private assistant evidence.
    for stage in meta.get('recovery_stages', []):
        name = str(stage.get('stage', stage.get('name','')))
        if label == 'stable_grasp' and 'stable' in name and 'grasp' in name:
            for key in ('end', 'end_time', 'end_sim_s', 'time'):
                value = stage.get(key)
                if isinstance(value,(float,int)) and math.isfinite(value):
                    return float(value)
    raise ValueError('missing '+label+' simulation time')


def accept_episode(*, rows, meta, status, dynamic, ctx, roll, episode_evidence):
    """Return complete sidecars only after all independent physical gates pass."""
    if not rows:
        raise ValueError('no expert frames')
    for key in ('task_success','physical_grasp_ok','placement_stable',
                'release_sequence_complete','released','target_collisions_enabled','inside_success_volume'):
        if status.get(key) is not True:
            raise ValueError('final physical acceptance failed: '+key)
    if status.get('holding_observed') or status.get('holding') or status.get('frozen') is not False:
        raise ValueError('final holding/frozen state invalidates recovery')
    for key in ('physical_grasp_verified','object_lifted_ever','bilateral_contact_ever',
                'release_sequence_complete','task_success'):
        if episode_evidence.get(key) is not True:
            raise ValueError('recorded physical evidence missing: '+key)
    if dynamic.get('finger_table_contact') or status.get('left_finger_touching_table') or status.get('right_finger_touching_table'):
        raise ValueError('finger/table contact invalidates recovery')
    if dynamic.get('samples',0) < 1 or dynamic.get('violations'):
        raise ValueError('missing or invalid dynamic status evidence')
    if dynamic.get('frame_samples') != len(rows) or dynamic.get('frame_violations'):
        raise ValueError('missing or invalid per-frame dynamic physics evidence')
    if not math.isfinite(roll) or abs(roll+90.) > 3.:
        raise ValueError('measured preclose FK roll must be -90 +/-3 degrees')
    if not meta.get('recovery_complete'):
        raise ValueError('expert recovery incomplete')
    if float(meta.get('withdrawal_rise_m',0.)) < .03:
        raise ValueError('successful post-release withdrawal requires >=3cm measured rise')
    if any(str(e.get('event','')).startswith('intervention_') for e in meta.get('events',[])):
        raise ValueError('failed-policy/intervention actions forbidden in expert recording')
    if ctx['candidate'].get('source_final_box_success') is not False:
        raise ValueError('successful source rollout excluded')
    labels,close,release = PROTOCOL.command_labels(rows,meta['events'])
    if len(rows)-release < 32:
        raise ValueError('at least32 release-tail frames required at20Hz')
    # The evaluator latches task_success/placement_stable. Current containment
    # and a newly measured quiet release tail are independently mandatory.
    samples=dynamic.get('status_samples',[])
    if not stable_release(status) or len(samples)<2:
        raise ValueError('missing current stable release tail evidence')
    latest=float(samples[-1]['simulation_time'])
    anchors=[i for i,x in enumerate(samples) if float(x['simulation_time'])<=latest-.5+1e-7]
    anchor=anchors[-1] if anchors else None
    if anchor is None:
        raise ValueError('missing stable release tail interval')
    tail_samples=samples[anchor:]
    tail_duration=latest-float(tail_samples[0]['simulation_time'])
    if (tail_duration<.5-1e-7 or latest<float(rows[-1]['simulation_time'])-.05
            or not all(stable_release(x['status']) for x in tail_samples)):
        raise ValueError('stable release tail must remain in box and quiet for >=0.5 seconds')
    start = _time(meta,('recovery_start_sim_s','recovery_start_time','recovery_start'),'recovery_start')
    stable = _time(meta,('stable_grasp_end_sim_s','stable_grasp_end_time','stable_grasp_end'),'stable_grasp')
    end = _time(meta,('recovery_end_sim_s','recovery_end_time','recovery_end'),'recovery_end')
    first,last = float(rows[0]['simulation_time']),float(rows[-1]['simulation_time'])
    if not first-.25 <= start <= end <= stable < float(rows[release]['simulation_time']) <= last:
        raise ValueError('recovery/stable-grasp interval must lie within expert recording before release')
    mask = [True]*len(rows)
    anchors16 = [i+16 <= len(rows) for i in range(len(rows))]
    anchors30 = [i+30 <= len(rows) for i in range(len(rows))]
    if not any(anchors30):
        raise ValueError('no complete30-action expert window')
    actual_close = next((i for i in range(close,release) if float(rows[i]['action'][-1]) > -.75),None)
    if actual_close is None or actual_close == 0:
        raise ValueError('no recorded actual close after open preclose state')
    correction = copy.deepcopy(meta)
    correction.update(accepted=True,task_id=ctx['task']['id'],task=copy.deepcopy(ctx['task']),
        split=ctx['task']['split'],plan_hash=ctx['plan_hash'],attempt_id=ctx['attempt_id'],
        source_type='actual_policy_push_recovery',source_identity=copy.deepcopy(ctx['source_identity']),
        source_grid=copy.deepcopy(ctx['task']['grid']),source_seed=ctx['candidate']['seed'],
        source_candidate=copy.deepcopy(ctx['candidate']),
        source_takeover_cup_expected_position_m=copy.deepcopy(ctx['candidate']['cup_xyz_m']),
        actual_takeover_cup_position_m=copy.deepcopy(meta.get('actual_cup_position_at_recording_m',ctx['candidate']['cup_xyz_m'])),
        actual_takeover_cup_orientation_xyzw=copy.deepcopy(meta.get('actual_cup_orientation_at_recording_xyzw',ctx['candidate'].get('cup_orientation_xyzw'))),
        measured_preclose_roll_deg=float(roll),close_command_frame=close,actual_close_action_frame=actual_close,
        preclose_fk_frame=actual_close-1,open_command_frame=release,
        recovery_intervals_s=[dict(start=start,end=stable)],
        final_status=copy.deepcopy(status),dynamic_samples=dynamic['samples'],
        dynamic_violations=copy.deepcopy(dynamic['violations']),
        dynamic_frame_samples=dynamic['frame_samples'],dynamic_frame_violations=copy.deepcopy(dynamic['frame_violations']),
        stable_release_tail_duration_s=tail_duration,stable_release_tail_samples=len(tail_samples),
        table_contact=False,physical_episode_evidence=copy.deepcopy(episode_evidence),
        eligible_action_horizon16_anchors=sum(anchors16),eligible_action_horizon30_anchors=sum(anchors30),
        recording_contains_failed_policy_actions=False)
    return dict(correction=correction,gripper_labels=dict(labels=labels,close_command_frame=close,
        open_command_frame=release,encoding='command-time intent; raw actual states/actions retained'),
        training_mask=dict(expert_frame_mask=mask,valid_action_horizon16_anchors=anchors16,
            valid_action_horizon30_anchors=anchors30,
            instruction='Expert-only recording starts after failed policy replay stops; all recorded actions are expert actions.'))


def make_recorder_class(Base, *, geometry=None, roll_fn=None):
    """Construct the private subclass without mutating production files."""
    class PushRecoveryRecorder(Base):
        def __init__(self):
            self.geometry = geometry if geometry is not None else _load('push_original_geometry',GEOMETRY_PATH).Geometry()
            self.roll_fn = roll_fn or _load('push_minus90_protocol',OLD_MINUS90/'protocol.py').tool_roll
            self.contact_stream = None
            self.dynamic_samples=0;self.dynamic_violations=[];self.contact=False
            self.physics_samples=[]
            self.frame_dynamic_samples=0;self.frame_dynamic_violations=[]
            self.ctx=None
            super().__init__()
            # Production already uses RLock; preserve re-entrant nested callbacks.
            self._lock = threading.RLock()
            self.require_task_success=True

        def _task_status_callback(self,message):
            with self._lock:
                super()._task_status_callback(message)
                if not self._recording:
                    return
                # Parse this exact callback payload; don't duplicate stale status
                # after a malformed JSON callback.
                try:status=json.loads(message.data)
                except (ValueError,TypeError):return
                stamp=self.get_clock().now().nanoseconds*1e-9
                self.dynamic_samples+=1
                self.physics_samples.append(dict(simulation_time=stamp,status=copy.deepcopy(status)))
                reason=physics_violation(status)
                self.contact |= bool(status.get('left_finger_touching_table') or status.get('right_finger_touching_table'))
                if reason and len(self.dynamic_violations)<50:
                    self.dynamic_violations.append(dict(simulation_time=stamp,reason=reason,status=status))
                if self.contact_stream is not None:
                    self.contact_stream.write(json.dumps(dict(simulation_time=stamp,status=status),ensure_ascii=False)+'\n')

        def _write_frame_locked(self):
            status=copy.deepcopy(self._selected_sample['status'])
            index=self._frame_count
            result=super()._write_frame_locked()
            self.frame_dynamic_samples+=1
            reason=physics_violation(status)
            self.contact |= bool(status.get('left_finger_touching_table') or status.get('right_finger_touching_table'))
            if reason and len(self.frame_dynamic_violations)<50:
                self.frame_dynamic_violations.append(dict(frame_index=index,reason=reason,status=status))
            return result

        def _start_callback(self,req,res):
            with self._lock:
                try:
                    if self._recording:raise ValueError('already recording')
                    self.ctx=context()
                    meta=json.loads(evidence_path(self.ctx).read_text())
                    validate_start(self.ctx,meta,self._task_status)
                    self.require_task_success=True
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    res.success=False;res.message='Push recovery start rejected: '+str(exc);return res
                self.dynamic_samples=0;self.dynamic_violations=[];self.contact=False
                self.physics_samples=[]
                self.frame_dynamic_samples=0;self.frame_dynamic_violations=[]
                res=super()._start_callback(req,res)
                if res.success:
                    self.takeover_status=copy.deepcopy(self._task_status)
                    if self.contact_stream is not None:self.contact_stream.close()
                    self.contact_stream=(self._episode_dir/'physics_status.jsonl').open('w',buffering=1)
                    self._episode_randomization=source_scene(self.ctx)
                    start=json.loads((self._episode_dir/'episode_start.json').read_text())
                    start.update(scene_randomization=self._episode_randomization,source_type='actual_policy_push_recovery',
                        source_grid=self.ctx['task']['grid'],source_identity=self.ctx['source_identity'],
                        expert_recording_start_simulation_time=self._start_sim_time,
                        failed_policy_prefix_recorded=False)
                    PROTOCOL.atomic_json(self._episode_dir/'episode_start.json',start)
                return res

        def _reject(self,res,error):
            res.success=False;res.message='Push recovery acceptance rejected: '+str(error)
            payload=dict(error=str(error),context=self.ctx,final_status=copy.deepcopy(self._task_status),
                dynamic_samples=self.dynamic_samples,dynamic_violations=self.dynamic_violations,
                frame_dynamic_samples=self.frame_dynamic_samples,frame_dynamic_violations=self.frame_dynamic_violations)
            if self.ctx is not None:
                PROTOCOL.atomic_json(evidence_path(self.ctx).with_suffix('.rejection.json'),payload)
            return res

        def _save_callback(self,req,res):
            with self._lock:
                try:
                    if not self._recording or self._frames_file is None:raise ValueError('no active expert recording')
                    if self._sync_failure:raise ValueError('synchronization failure: '+self._sync_failure)
                    self._frames_file.flush()
                    if self.contact_stream is not None:self.contact_stream.flush()
                    rows=[json.loads(line) for line in (self._episode_dir/'frames.jsonl').read_text().splitlines()]
                    if len(rows)<self.minimum_frames:raise ValueError('insufficient recorded frames')
                    meta=json.loads(evidence_path(self.ctx).read_text())
                    meta['actual_cup_position_at_recording_m']=copy.deepcopy(self.takeover_status.get('object_position_m'))
                    meta['actual_cup_orientation_at_recording_xyzw']=copy.deepcopy(self.takeover_status.get('object_orientation_xyzw'))
                    meta['actual_takeover_status']=copy.deepcopy(self.takeover_status)
                    _,close,release=PROTOCOL.command_labels(rows,meta['events'])
                    actual_close=next(i for i in range(close,release) if float(rows[i]['action'][-1])>-.75)
                    if actual_close<=0:raise ValueError('missing open preclose frame')
                    _,rotation=self.geometry.link_fk(rows[actual_close-1]['observation.state'][:6])
                    roll=float(self.roll_fn(rotation))
                    # Include final status in exactly the same cumulative evidence
                    # the production save gate consumes.
                    evidence=copy.deepcopy(self._episode_evidence)
                    for key,status_key in [('physical_grasp_verified','physical_grasp_ok'),('object_lifted_ever','lifted'),('bilateral_contact_ever','bilateral_contact'),('release_sequence_complete','release_sequence_complete'),('task_success','task_success')]:
                        evidence[key]=bool(evidence.get(key) or self._task_status.get(status_key))
                    payload=accept_episode(rows=rows,meta=meta,status=self._task_status,
                        dynamic=dict(samples=self.dynamic_samples,violations=self.dynamic_violations,finger_table_contact=self.contact,
                            frame_samples=self.frame_dynamic_samples,frame_violations=self.frame_dynamic_violations,status_samples=self.physics_samples),
                        ctx=self.ctx,roll=roll,episode_evidence=evidence)
                    episode_dir=self._episode_dir
                    for name,value in payload.items():PROTOCOL.atomic_json(episode_dir/(name+'.json'),value)
                    self.require_task_success=True
                    result=super()._save_callback(req,res)
                    if not result.success:
                        payload['correction']['accepted']=False
                        PROTOCOL.atomic_json(episode_dir/'correction.json',payload['correction'])
                        return self._reject(result,result.message)
                    if self.contact_stream is not None:
                        self.contact_stream.flush();self.contact_stream.close();self.contact_stream=None
                    PROTOCOL.atomic_json(evidence_path(self.ctx),payload['correction'])
                    return result
                except (OSError,ValueError,KeyError,TypeError,StopIteration,RuntimeError) as exc:
                    return self._reject(res,exc)

        def _discard_callback(self,req,res):
            with self._lock:
                if self.contact_stream is not None:
                    self.contact_stream.flush();self.contact_stream.close();self.contact_stream=None
                return super()._discard_callback(req,res)

        def destroy_node(self):
            with self._lock:
                if self.contact_stream is not None:
                    self.contact_stream.flush();self.contact_stream.close();self.contact_stream=None
            return super().destroy_node()
    return PushRecoveryRecorder


def main(args=None):
    import z1_isaac_bridge.episode_recorder as production
    sys.path.insert(0,'/home/lp531/z1_queue_logs/cup_reproduce_20hz_20261031')
    from common import guarded_recorder
    production.EpisodeRecorder=make_recorder_class(guarded_recorder(production.EpisodeRecorder,"recovery"))
    production.main(args=args)


recorder_main=main
if __name__=='__main__':main()
