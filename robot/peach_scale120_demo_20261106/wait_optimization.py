"""Conservative reuse of peach preplans and immediately adjacent grasp probes.

Only the basic peach recorder uses this mixin. It never changes trajectory
stops, closing, contact predicates, controller tolerances, or release checks.
"""
import copy
import math
import json
import time


def distance(a, b):
    return math.dist(a, b)


def angle(a, b):
    norms = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    if not math.isfinite(norms) or norms <= 0:
        return math.inf
    dot = abs(sum(x*y for x,y in zip(a,b)))/norms
    return 2*math.acos(min(1., dot))


class PeachWaitOptimization:
    def _clear_peach_wait_caches(self):
        with self._lock:
            self._peach_preplanned_transfer = None
            self._peach_adjacent_probe = None
            self._peach_wait_epoch = getattr(self, '_peach_wait_epoch', 0)+1

    def _target_callback(self, message):
        self._clear_peach_wait_caches()
        return super()._target_callback(message)

    def _reset_assistant_service(self, *args):
        self._clear_peach_wait_caches()
        return super()._reset_assistant_service(*args)

    def _stop_assistant_service(self, *args):
        self._clear_peach_wait_caches()
        return super()._stop_assistant_service(*args)

    def _task_status_callback(self, message):
        try:
            parsed = json.loads(message.data)
        except (AttributeError, ValueError, TypeError):
            return super()._task_status_callback(message)
        if not isinstance(parsed, dict):
            return super()._task_status_callback(message)
        result = super()._task_status_callback(message)
        with self._lock:
            self._task_status_seen_at = time.monotonic()
        return result

    def _scene_snapshot(self):
        with self._lock:
            return copy.deepcopy((self._object_poses, self._object_orientations))

    def _scene_unchanged(self, snapshot, tolerance):
        positions, orientations = snapshot
        current, current_orientations = self._scene_snapshot()
        if not positions or set(positions) != set(current):
            return False
        now = time.monotonic()
        for name, (_, position) in positions.items():
            stamp, actual = current[name]
            if not 0 <= now-stamp <= getattr(self, 'object_timeout', 2.):
                return False
            separation = distance(position, actual)
            if not math.isfinite(separation) or separation > tolerance:
                return False
            if name not in orientations or name not in current_orientations:
                return False
            orientation_stamp = current_orientations[name][0]
            if not 0 <= now-orientation_stamp <= getattr(self, 'object_timeout', 2.):
                return False
            if angle(orientations[name][1], current_orientations[name][1]) > .035:
                return False
        return True

    def _generate_plan(self):
        self._clear_peach_wait_caches()
        epoch = self._peach_wait_epoch
        self._peach_wait_prechecks = []
        self._peach_wait_selecting = True
        try:
            result = super()._generate_plan()
        finally:
            self._peach_wait_selecting = False
        with self._lock:
            candidate = self._grasp_candidate
            if self.target_object == 'peach' and candidate is not None and epoch == self._peach_wait_epoch:
                for record in reversed(self._peach_wait_prechecks):
                    if (distance(record['ee'], candidate['position']) < 1e-6
                            and angle(record['orientation'], candidate['orientation']) < 1e-6):
                        record['owner'] = candidate
                        self._peach_preplanned_transfer = record
                        break
            self._peach_wait_prechecks = []
        return result

    def _plan_apple_transfer(self, origin, ee_origin, orientation, height, start_state=None):
        self._raise_if_stop_requested()
        if getattr(self, '_peach_wait_selecting', False):
            scene = self._scene_snapshot()
            created = time.monotonic()
            route = super()._plan_apple_transfer(origin, ee_origin, orientation, height, start_state=start_state)
            if route:
                self._peach_wait_prechecks.append(dict(route=copy.deepcopy(route),scene=scene,
                    created=created,ee=tuple(ee_origin),origin=tuple(origin),
                    orientation=tuple(orientation),height=float(height)))
            return route
        with self._lock:
            record = getattr(self, '_peach_preplanned_transfer', None)
            self._peach_preplanned_transfer = None  # Single use, including failed validation.
            owner = self._grasp_candidate
            joints = dict(self._joint_positions)
        if record and start_state is None and record['owner'] is owner and self.target_object == 'peach':
            path = record['route'][0]['trajectory'].joint_trajectory
            start = path.points[0].positions if path.points else []
            valid_start = (bool(start) and len(start) == len(path.joint_names)
                           and all(name in joints and abs(joints[name]-value) <= .03
                                   for name,value in zip(path.joint_names,start)))
            if (0 <= time.monotonic()-record['created'] <= 30.
                    and abs(height-record['height']) < .001
                    and distance(origin,record['origin']) <= .005
                    and distance(ee_origin,record['ee']) <= .005
                    and angle(orientation,record['orientation']) <= .035
                    and valid_start and self._scene_unchanged(record['scene'], .003)):
                self.get_logger().info('桃子：复用抓取前已验证的完整搬运计划；起点及场景复核通过')
                return record['route']
            self.get_logger().info('桃子：预检查计划已失效，按实际夹持状态重新规划')
        return super()._plan_apple_transfer(origin, ee_origin, orientation, height, start_state=start_state)

    def _require_loaded_physical_grasp(self, target_object, envelope, stage, **kwargs):
        self._raise_if_stop_requested()
        with self._lock:
            proof = getattr(self, '_peach_adjacent_probe', None)
            if stage == '桃子已规划路线执行前':
                self._peach_adjacent_probe = None
            joints = dict(self._joint_positions)
            status = dict(self._task_status)
            status_time = getattr(self, '_task_status_seen_at', -math.inf)
            owner = self._grasp_candidate
        if target_object == 'peach' and stage == '桃子已规划路线执行前' and proof:
            if (self.target_object == 'peach' and proof['owner'] is owner
                    and proof['envelope'] == envelope and proof['kwargs'] == kwargs
                    and 0 <= time.monotonic()-proof['created'] <= .25
                    and 0 <= time.monotonic()-status_time <= .2
                    and status.get('bilateral_contact') is True
                    and status.get('physical_grasp_ok') is True
                    and not status.get('left_finger_touching_table', True)
                    and not status.get('right_finger_touching_table', True)
                    and not status.get('failure_reason')
                    and set(joints) == set(proof['joints'])
                    and all(abs(joints[k]-proof['joints'][k]) <= .003 for k in joints)
                    and self._scene_unchanged(proof['scene'], .003)):
                self.get_logger().info('桃子：相邻搬运段复用刚完成的夹持验收，实时接触仍有效')
                return copy.deepcopy(proof['probe'])
        # Invalidate before any new probe, so an exception cannot leave a valid token.
        with self._lock:
            self._peach_adjacent_probe = None
        probe = super()._require_loaded_physical_grasp(target_object,envelope,stage,**kwargs)
        if target_object == 'peach' and stage == '桃子已规划路线执行后':
            with self._lock:
                self._peach_adjacent_probe = dict(owner=self._grasp_candidate,probe=copy.deepcopy(probe),
                    envelope=copy.deepcopy(envelope),kwargs=copy.deepcopy(kwargs),created=time.monotonic(),
                    joints=dict(self._joint_positions),scene=self._scene_snapshot())
        return probe
