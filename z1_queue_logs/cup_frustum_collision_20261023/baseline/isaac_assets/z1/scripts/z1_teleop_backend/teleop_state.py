import threading

import numpy as np
from isaacsim.core.utils.types import ArticulationAction

from .config import (
    ARM_JOINTS,
    DEFAULT_SPEED_SCALE,
    MAX_SAFE_SPEED_RAD_S,
    MIN_LIMIT_MARGIN,
    MOVEIT_ARM_JOINTS,
    SAFE_HOME,
)
from .safety import clamp


class TeleopState:
    def __init__(self, robot, urdf_limits, real_arm=None):
        self.robot = robot
        self.controller = robot.get_articulation_controller()
        self.urdf_limits = urdf_limits
        self.real_arm = real_arm
        self.lock = threading.Lock()
        self.enabled = False
        self.estop = False
        self.external_arm_control = False
        self.step_count = 0
        self.clamp_count = 0
        self.last_message = "未使能"
        self.speed_scale = DEFAULT_SPEED_SCALE

        self.dof_names = list(robot.dof_names)
        self.dof_name_to_index = {name: idx for idx, name in enumerate(self.dof_names)}
        self.control_indices = [
            idx for idx, name in enumerate(self.dof_names) if name in ARM_JOINTS and name in urdf_limits
        ]
        self.control_index_set = set(self.control_indices)
        if not self.control_indices:
            raise RuntimeError("没有找到可控的 Z1 DOF。")

        current = self._read_robot_positions()
        self.current_positions = current.copy()
        self.targets = current.copy()
        for idx in self.control_indices:
            name = self.dof_names[idx]
            low, high = self.safe_bounds(name)
            self.targets[idx] = clamp(self.targets[idx], low, high)
        self.max_speed = self._speed_vector(self.speed_scale)

    def _read_robot_positions(self):
        positions = self.robot.get_joint_positions()
        if positions is None:
            return np.zeros(self.robot.num_dof, dtype=np.float64)
        return np.asarray(positions, dtype=np.float64).copy()

    def _current_positions(self):
        return self.current_positions.copy()

    def safe_bounds(self, joint_name):
        info = self.urdf_limits[joint_name]
        return info["lower"] + MIN_LIMIT_MARGIN, info["upper"] - MIN_LIMIT_MARGIN

    def _set_speed_locked(self, speed_scale):
        self.speed_scale = clamp(float(speed_scale), 0.05, 1.0)
        self.max_speed = self._speed_vector(self.speed_scale)

    def _speed_vector(self, speed_scale):
        speed_scale = clamp(float(speed_scale), 0.05, 1.0)
        speeds = np.zeros(self.robot.num_dof, dtype=np.float64)
        for idx in self.control_indices:
            name = self.dof_names[idx]
            urdf_speed = self.urdf_limits[name]["velocity"]
            speeds[idx] = min(urdf_speed, MAX_SAFE_SPEED_RAD_S) * speed_scale
        return speeds

    def _real_targets_locked(self):
        targets = {}
        for joint_name in ARM_JOINTS:
            idx = self.dof_name_to_index.get(joint_name)
            if idx is None or idx not in self.control_index_set:
                continue
            targets[joint_name] = float(self.targets[idx])
        return targets

    def _send_real_targets(self, targets, speed_scale, force=False):
        if self.real_arm is not None:
            self.real_arm.send_targets(targets, speed_scale, force=force)

    def set_speed(self, speed_scale):
        with self.lock:
            self._set_speed_locked(speed_scale)
            self.last_message = f"速度倍率已更新: {self.speed_scale:.2f}"
            targets = self._real_targets_locked()
            speed = self.speed_scale
            should_sync = self.enabled and not self.estop
            snapshot = self.snapshot_locked()
        if should_sync:
            self._send_real_targets(targets, speed, force=True)
        return snapshot

    def set_external_arm_control(self, enabled, message=None):
        with self.lock:
            self.external_arm_control = bool(enabled)
            self.targets = self._current_positions()
            if message is not None:
                self.last_message = message
            return self.snapshot_locked()

    def moveit_start_state(self):
        with self.lock:
            current = self._read_robot_positions()
            joint_names = []
            positions = []
            for name in MOVEIT_ARM_JOINTS:
                idx = self.dof_name_to_index.get(name)
                if idx is not None:
                    joint_names.append(name)
                    positions.append(float(current[idx]))
            return {"joint_names": joint_names, "positions": positions}

    def set_enabled(self, enabled, speed_scale=DEFAULT_SPEED_SCALE):
        with self.lock:
            self.targets = self._current_positions()
            self._set_speed_locked(speed_scale)
            self.enabled = bool(enabled)
            self.estop = False if enabled else self.estop
            self.last_message = "已使能" if enabled else "已禁用"
            targets = self._real_targets_locked()
            speed = self.speed_scale
            snapshot = self.snapshot_locked()
        if self.real_arm is not None:
            if enabled:
                self.real_arm.enable(targets, speed)
            else:
                self.real_arm.disable()
        return snapshot

    def estop_now(self):
        with self.lock:
            self.targets = self._current_positions()
            self.enabled = False
            self.estop = True
            self.last_message = "急停已触发，机械臂保持当前位置"
            snapshot = self.snapshot_locked()
        if self.real_arm is not None:
            self.real_arm.stop()
        return snapshot

    def hold(self):
        with self.lock:
            self.targets = self._current_positions()
            self.last_message = "目标已同步到当前位置"
            snapshot = self.snapshot_locked()
        if self.real_arm is not None:
            self.real_arm.hold()
        return snapshot

    def home(self, speed_scale=DEFAULT_SPEED_SCALE):
        with self.lock:
            for idx in self.control_indices:
                name = self.dof_names[idx]
                if name in SAFE_HOME:
                    low, high = self.safe_bounds(name)
                    self.targets[idx] = clamp(SAFE_HOME[name], low, high)
            self._set_speed_locked(speed_scale)
            self.enabled = True
            self.estop = False
            self.last_message = "正在回安全位"
            targets = self._real_targets_locked()
            speed = self.speed_scale
            snapshot = self.snapshot_locked()
        if self.real_arm is not None:
            self.real_arm.enable(targets, speed)
        return snapshot

    def set_named_targets(self, targets_by_name, speed_scale=DEFAULT_SPEED_SCALE, message="目标已更新"):
        with self.lock:
            for joint_name, value in targets_by_name.items():
                idx = self.dof_name_to_index.get(joint_name)
                if idx is None or joint_name not in self.urdf_limits:
                    continue
                if idx not in self.control_index_set:
                    continue
                low, high = self.safe_bounds(joint_name)
                clipped = clamp(float(value), low, high)
                if clipped != float(value):
                    self.clamp_count += 1
                self.targets[idx] = clipped
            self._set_speed_locked(speed_scale)
            self.enabled = True
            self.estop = False
            self.last_message = message
            targets = self._real_targets_locked()
            speed = self.speed_scale
            snapshot = self.snapshot_locked()
        self._send_real_targets(targets, speed, force=True)
        return snapshot

    def apply_joint_positions(self, positions_by_name, message=None):
        with self.lock:
            current = self._read_robot_positions()
            command = current.copy()
            for joint_name, value in positions_by_name.items():
                idx = self.dof_name_to_index.get(joint_name)
                if idx is None or joint_name not in self.urdf_limits:
                    continue
                if idx not in self.control_index_set:
                    continue
                low, high = self.safe_bounds(joint_name)
                clipped = clamp(float(value), low, high)
                if clipped != float(value):
                    self.clamp_count += 1
                command[idx] = clipped
                self.targets[idx] = clipped
            self.controller.apply_action(ArticulationAction(joint_positions=command))
            self.current_positions = command.copy()
            self.enabled = True
            self.estop = False
            if message is not None:
                self.last_message = message
            targets = self._real_targets_locked()
            speed = self.speed_scale
            snapshot = self.snapshot_locked()
        self._send_real_targets(targets, speed, force=True)
        return snapshot

    def set_joint(self, joint_name, value, speed_scale=DEFAULT_SPEED_SCALE):
        with self.lock:
            idx = self._joint_index(joint_name)
            low, high = self.safe_bounds(joint_name)
            clipped = clamp(float(value), low, high)
            if clipped != float(value):
                self.clamp_count += 1
                self.last_message = f"{joint_name} 目标超限，已夹到安全范围"
            else:
                self.last_message = f"{joint_name} 目标已更新"
            self.targets[idx] = clipped
            self._set_speed_locked(speed_scale)
            targets = self._real_targets_locked()
            speed = self.speed_scale
            should_sync = self.enabled and not self.estop
            snapshot = self.snapshot_locked()
        if should_sync:
            self._send_real_targets(targets, speed, force=True)
        return snapshot

    def jog(self, joint_name, delta, speed_scale=DEFAULT_SPEED_SCALE):
        with self.lock:
            idx = self._joint_index(joint_name)
            low, high = self.safe_bounds(joint_name)
            requested = self.targets[idx] + float(delta)
            clipped = clamp(requested, low, high)
            if clipped != requested:
                self.clamp_count += 1
                self.last_message = f"{joint_name} 步进触到限位，已停止在安全范围内"
            else:
                self.last_message = f"{joint_name} 步进 {float(delta):.3f} rad"
            self.targets[idx] = clipped
            self._set_speed_locked(speed_scale)
            targets = self._real_targets_locked()
            speed = self.speed_scale
            should_sync = self.enabled and not self.estop
            snapshot = self.snapshot_locked()
        if should_sync:
            self._send_real_targets(targets, speed, force=True)
        return snapshot

    def _joint_index(self, joint_name):
        idx = self.dof_name_to_index.get(joint_name)
        if idx is None:
            raise ValueError(f"未知关节: {joint_name}")
        if idx not in self.control_index_set:
            raise ValueError(f"该关节未开放遥操作: {joint_name}")
        return idx

    def step(self, dt):
        targets = None
        speed = None
        with self.lock:
            self.step_count += 1
            self.current_positions = self._read_robot_positions()
            if not self.enabled or self.estop:
                return
            current = self.current_positions.copy()
            command = current.copy()
            max_delta = np.maximum(self.max_speed * float(dt), 0.001)
            for idx in self.control_indices:
                name = self.dof_names[idx]
                low, high = self.safe_bounds(name)
                target = clamp(self.targets[idx], low, high)
                if target != self.targets[idx]:
                    self.targets[idx] = target
                    self.clamp_count += 1
                delta = clamp(target - current[idx], -max_delta[idx], max_delta[idx])
                command[idx] = clamp(current[idx] + delta, low, high)
            self.controller.apply_action(ArticulationAction(joint_positions=command))
            self.current_positions = command.copy()
            targets = self._real_targets_locked()
            speed = self.speed_scale
        self._send_real_targets(targets, speed, force=False)

    def snapshot(self):
        with self.lock:
            return self.snapshot_locked()

    def snapshot_locked(self):
        current = self._current_positions()
        joints = []
        for idx in self.control_indices:
            name = self.dof_names[idx]
            info = self.urdf_limits[name]
            joints.append(
                {
                    "index": idx,
                    "name": name,
                    "position": float(current[idx]),
                    "target": float(self.targets[idx]),
                    "lower": info["lower"] + MIN_LIMIT_MARGIN,
                    "upper": info["upper"] - MIN_LIMIT_MARGIN,
                    "velocity": min(info["velocity"], MAX_SAFE_SPEED_RAD_S),
                }
            )
        return {
            "enabled": self.enabled,
            "estop": self.estop,
            "step_count": self.step_count,
            "clamp_count": self.clamp_count,
            "speed_scale": self.speed_scale,
            "external_arm_control": self.external_arm_control,
            "message": self.last_message,
            "joints": joints,
        }
