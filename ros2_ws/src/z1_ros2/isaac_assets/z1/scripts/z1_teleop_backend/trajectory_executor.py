import threading
import time

from .config import MOVEIT_LOCKED_JOINTS, MOVEIT_WARN_JOINT_DELTAS


class TrajectoryExecutor:
    def __init__(self, teleop):
        self.teleop = teleop
        self.lock = threading.Lock()
        self.active = False
        self.phase = "idle"
        self.joint_names = []
        self.points = []
        self.index = 0
        self.start_time = 0.0
        self.last_error = ""
        self.completed_phase = ""
        self.last_diagnostics = {}

    def start(self, phase, trajectory):
        joint_names = trajectory.get("joint_names") or []
        points = trajectory.get("points") or []
        if not joint_names or not points:
            raise RuntimeError("MoveIt 返回的关节轨迹为空")
        diagnostics = self._trajectory_diagnostics(joint_names, points)
        violations = diagnostics.get("joint_delta_violations") or {}
        if violations:
            raise RuntimeError(f"MoveIt 轨迹关节变化异常: {violations}")
        points = self._lock_configured_joints(joint_names, points)
        with self.lock:
            self.active = True
            self.phase = phase
            self.joint_names = list(joint_names)
            self.points = list(points)
            self.index = 0
            self.start_time = time.monotonic()
            self.last_error = ""
            self.completed_phase = ""
            self.last_diagnostics = diagnostics

    def _trajectory_diagnostics(self, joint_names, points):
        start_state = self.teleop.moveit_start_state()
        start_positions = dict(zip(start_state["joint_names"], start_state["positions"]))
        max_delta_by_joint = {}
        violations = {}
        for joint_name, limit in MOVEIT_WARN_JOINT_DELTAS.items():
            if joint_name not in joint_names or joint_name not in start_positions:
                continue
            idx = joint_names.index(joint_name)
            deltas = [
                abs(float((point.get("positions") or [])[idx]) - float(start_positions[joint_name]))
                for point in points
                if len(point.get("positions") or []) > idx
            ]
            max_delta = max(deltas, default=0.0)
            max_delta_by_joint[joint_name] = max_delta
            if max_delta > float(limit):
                violations[joint_name] = {
                    "max_delta_rad": max_delta,
                    "limit_rad": float(limit),
                    "start_rad": float(start_positions[joint_name]),
                }
        return {
            "max_delta_by_joint": max_delta_by_joint,
            "joint_delta_violations": violations,
        }

    def _lock_configured_joints(self, joint_names, points):
        locked = [name for name in MOVEIT_LOCKED_JOINTS if name in joint_names]
        if not locked:
            return list(points)
        start_state = self.teleop.moveit_start_state()
        start_positions = dict(zip(start_state["joint_names"], start_state["positions"]))
        locked_values = {name: start_positions[name] for name in locked if name in start_positions}
        if not locked_values:
            return list(points)
        locked_points = []
        for point in points:
            updated = dict(point)
            positions = list(point.get("positions") or [])
            for joint_name, value in locked_values.items():
                positions[joint_names.index(joint_name)] = float(value)
            updated["positions"] = positions
            locked_points.append(updated)
        return locked_points

    def cancel(self, message="轨迹执行已取消"):
        with self.lock:
            self.active = False
            self.phase = "idle"
            self.last_error = message

    def step(self):
        with self.lock:
            if not self.active:
                return
            phase = self.phase
            elapsed = time.monotonic() - self.start_time
            while self.index + 1 < len(self.points) and float(self.points[self.index + 1].get("time_from_start_s", 0.0)) <= elapsed:
                self.index += 1
            point = self.points[self.index]
            positions = point.get("positions") or []
            if len(positions) != len(self.joint_names):
                self.active = False
                self.last_error = "轨迹点关节数量与 joint_names 不匹配"
                return
            positions_by_name = dict(zip(self.joint_names, positions))
            done = self.index >= len(self.points) - 1 and elapsed >= float(point.get("time_from_start_s", 0.0))

        self.teleop.apply_joint_positions(positions_by_name, message=f"Isaac 正在执行 {phase} 轨迹")

        if done:
            with self.lock:
                self.active = False
                self.completed_phase = phase
                self.phase = "idle"

    def consume_completed(self, phase):
        with self.lock:
            if self.completed_phase != phase:
                return False
            self.completed_phase = ""
            return True

    def snapshot(self):
        with self.lock:
            return {
                "active": self.active,
                "phase": self.phase,
                "point_index": self.index,
                "point_count": len(self.points),
                "joint_names": list(self.joint_names),
                "locked_joints": list(MOVEIT_LOCKED_JOINTS),
                "diagnostics": dict(self.last_diagnostics),
                "error": self.last_error,
            }
