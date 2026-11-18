"""Expose ros2_control-compatible trajectory actions backed by Isaac JointState commands."""

from __future__ import annotations

import math
import json
import threading
import time
from typing import Dict, Iterable, List, Sequence

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINTS = ("jointGripper",)
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS
JOINT_LIMITS = {
    "joint1": (-math.radians(150.0), math.radians(150.0)),
    "joint2": (0.0, math.radians(170.0)),
    "joint3": (-math.radians(165.0), 0.0),
    "joint4": (-math.radians(87.0), math.radians(87.0)),
    "joint5": (-math.radians(77.0), math.radians(77.0)),
    "joint6": (-math.radians(160.0), math.radians(160.0)),
    "jointGripper": (-math.radians(90.0), 0.0),
}


def duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1.0e-9


def interpolate(start: Sequence[float], end: Sequence[float], ratio: float) -> List[float]:
    ratio = max(0.0, min(1.0, ratio))
    return [a + (b - a) * ratio for a, b in zip(start, end)]


class TrajectoryBridge(Node):
    """Translate FollowJointTrajectory goals into Isaac Sim JointState commands."""

    def __init__(self) -> None:
        super().__init__("z1_isaac_trajectory_bridge")
        self.declare_parameter("command_topic", "/isaac_joint_commands")
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("command_rate", 100.0)
        self.declare_parameter("state_timeout", 3.0)
        self.declare_parameter("goal_tolerance", 0.012)
        self.declare_parameter("goal_time_tolerance", 3.0)
        self.declare_parameter("arm_goal_stable_duration", 0.20)
        self.declare_parameter("joint_limit_tolerance", 0.002)
        self.declare_parameter(
            "servo_command_topic", "/smolvla/servo_joint_trajectory"
        )
        self.declare_parameter("servo_max_joint_step_rad", 0.10)
        self.declare_parameter("path_error_limit_rad", 0.08)
        self.declare_parameter("path_error_duration_s", 0.20)
        self.declare_parameter("path_error_hard_limit_rad", 0.20)
        self.path_error_limit = float(self.get_parameter("path_error_limit_rad").value)
        self.path_error_duration = float(self.get_parameter("path_error_duration_s").value)
        self.path_error_hard_limit = float(self.get_parameter("path_error_hard_limit_rad").value)
        if not (0 < self.path_error_limit <= self.path_error_hard_limit < math.inf
                and 0 < self.path_error_duration < math.inf):
            raise ValueError("Invalid path error thresholds")

        command_topic = self.get_parameter("command_topic").value
        state_topic = self.get_parameter("state_topic").value
        self.command_rate = float(self.get_parameter("command_rate").value)
        self.state_timeout = float(self.get_parameter("state_timeout").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.goal_time_tolerance = float(self.get_parameter("goal_time_tolerance").value)
        self.arm_goal_stable_duration = float(
            self.get_parameter("arm_goal_stable_duration").value
        )
        self.joint_limit_tolerance = float(
            self.get_parameter("joint_limit_tolerance").value
        )
        servo_command_topic = str(
            self.get_parameter("servo_command_topic").value
        )
        self.servo_max_joint_step = float(
            self.get_parameter("servo_max_joint_step_rad").value
        )

        self._callback_group = ReentrantCallbackGroup()
        self._state_lock = threading.RLock()
        self._controller_locks = {
            "arm": threading.Lock(),
            "gripper": threading.Lock(),
        }
        self._actual: Dict[str, float] = {}
        self._command: Dict[str, float] = {}
        self._last_state_time = 0.0
        self._joint_state_times = {}
        self._stop_generation = 0
        self._servo_stream_enabled = False
        self._policy_control_active = False
        self._recording = False
        self._recording_status_time = 0.0
        self._motion_fault = ""
        self._fault_hold = {}
        self._scene_reset_active = False
        self._contact_time = 0.0
        self._touching_table = False
        self._path_error_since = None

        self._publisher = self.create_publisher(JointState, command_topic, 10)
        self.create_service(
            Trigger, "/smolvla/begin_scene_reset", self._begin_scene_reset_callback,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            String, "/smolvla/status", self._contact_callback, 10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            Bool,
            "/smolvla/policy_control_active",
            self._policy_control_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger, "/smolvla/clear_motion_fault", self._clear_fault_callback,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            JointState,
            state_topic,
            self._state_callback,
            10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            JointTrajectory,
            servo_command_topic,
            self._servo_trajectory_callback,
            10,
            callback_group=self._callback_group,
        )
        self._arm_server = self._create_server(
            "/joint_trajectory_controller/follow_joint_trajectory",
            ARM_JOINTS,
            "arm",
        )
        self._gripper_server = self._create_server(
            "/gripper_trajectory_controller/follow_joint_trajectory",
            GRIPPER_JOINTS,
            "gripper",
        )
        self.create_service(
            Trigger,
            "/smolvla/stop_motion",
            self._stop_motion_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            SetBool,
            "/smolvla/enable_servo_stream",
            self._enable_servo_stream_callback,
            callback_group=self._callback_group,
        )
        recording_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/smolvla/recording_status", self._recording_status_callback,
            recording_qos, callback_group=self._callback_group,
        )
        self.create_timer(0.02, self._motion_watchdog, callback_group=self._callback_group)
        self.get_logger().info(
            "Isaac trajectory bridge ready: %s -> %s" % (state_topic, command_topic)
        )

    def _create_server(self, name: str, joints: Sequence[str], controller: str):
        return ActionServer(
            self,
            FollowJointTrajectory,
            name,
            goal_callback=lambda request: self._goal_callback(request, joints),
            cancel_callback=self._cancel_callback,
            execute_callback=lambda goal: self._execute(goal, joints, controller),
            callback_group=self._callback_group,
        )

    def _state_callback(self, message: JointState) -> None:
        with self._state_lock:
            for name, position in zip(message.name, message.position):
                if name in ALL_JOINTS:
                    value = float(position)
                    if not math.isfinite(value):
                        self._trip_fault("Non-finite joint feedback")
                        return
                    self._actual[name] = value
                    self._joint_state_times[name] = time.monotonic()
                    self._command.setdefault(name, value)
            if all(name in self._joint_state_times for name in ALL_JOINTS):
                self._last_state_time = min(self._joint_state_times.values())

    def _goal_callback(self, request, expected_joints: Sequence[str]):
        with self._state_lock:
            if self._policy_control_active:
                self.get_logger().error(
                    "Rejected trajectory while SmolVLA policy owns control"
                )
                return GoalResponse.REJECT
            if self._motion_fault:
                self.get_logger().error("Motion fault latched: " + self._motion_fault)
                return GoalResponse.REJECT
            servo_stream_enabled = self._servo_stream_enabled
        if expected_joints == ARM_JOINTS and servo_stream_enabled:
            self.get_logger().error(
                "Rejected global arm trajectory while Servo stream is enabled"
            )
            return GoalResponse.REJECT
        trajectory = request.trajectory
        if set(trajectory.joint_names) != set(expected_joints):
            self.get_logger().error(
                "Rejected trajectory joints %s; expected %s"
                % (list(trajectory.joint_names), list(expected_joints))
            )
            return GoalResponse.REJECT
        if not trajectory.points:
            self.get_logger().error("Rejected empty trajectory")
            return GoalResponse.REJECT
        previous_time = -1.0
        for point in trajectory.points:
            point_time = duration_seconds(point.time_from_start)
            if len(point.positions) != len(expected_joints) or point_time < previous_time:
                self.get_logger().error("Rejected malformed or non-monotonic trajectory")
                return GoalResponse.REJECT
            for name, position in zip(trajectory.joint_names, point.positions):
                lower, upper = JOINT_LIMITS[name]
                if (
                    not math.isfinite(position)
                    or position < lower - self.joint_limit_tolerance
                    or position > upper + self.joint_limit_tolerance
                ):
                    self.get_logger().error(
                        "Rejected %s=%.4f outside [%.4f, %.4f] with %.4f tolerance"
                        % (
                            name,
                            position,
                            lower,
                            upper,
                            self.joint_limit_tolerance,
                        )
                    )
                    return GoalResponse.REJECT
            previous_time = point_time
        return GoalResponse.ACCEPT

    def _enable_servo_stream_callback(self, request, response):
        """Exclusively enable bounded Servo commands after global motion."""
        if request.data:
            with self._state_lock:
                if self._policy_control_active:
                    response.success = False
                    response.message = "SmolVLA policy currently owns control"
                    return response
            if self._controller_locks["arm"].locked():
                response.success = False
                response.message = "全局机械臂轨迹仍在执行，拒绝启用 Servo"
                return response
            with self._state_lock:
                if self._motion_fault:
                    response.success = False
                    response.message = self._motion_fault
                    return response
                if not all(name in self._actual for name in ARM_JOINTS):
                    response.success = False
                    response.message = "关节状态不完整，拒绝启用 Servo"
                    return response
                self._servo_stream_enabled = True
            response.success = True
            response.message = "Servo 关节流已独占启用"
            return response

        with self._state_lock:
            self._servo_stream_enabled = False
            hold = [self._actual.get(name) for name in ARM_JOINTS]
        if all(value is not None for value in hold):
            self._publish_command(ARM_JOINTS, hold)
        response.success = True
        response.message = "Servo 关节流已禁用并保持当前位置"
        return response

    def _policy_control_callback(self, message: Bool) -> None:
        """Prevent action and Servo publishers from racing policy rollout."""
        with self._state_lock:
            active = bool(message.data)
            if active and not self._policy_control_active:
                self._stop_generation += 1
                self._servo_stream_enabled = False
            self._policy_control_active = active

    def _servo_trajectory_callback(self, message: JointTrajectory) -> None:
        """Forward only explicitly enabled, small Servo joint increments."""
        with self._state_lock:
            enabled = self._servo_stream_enabled
            policy_active = self._policy_control_active
            actual = [self._actual.get(name) for name in ARM_JOINTS]
        if policy_active or not enabled or self._controller_locks["arm"].locked():
            return
        if self._check_motion(ARM_JOINTS):
            return
        if set(message.joint_names) != set(ARM_JOINTS) or not message.points:
            self.get_logger().error("Rejected malformed Servo joint trajectory")
            return
        point = message.points[-1]
        if len(point.positions) != len(ARM_JOINTS):
            self.get_logger().error("Rejected incomplete Servo joint target")
            return
        target_by_name = dict(zip(message.joint_names, point.positions))
        targets = [float(target_by_name[name]) for name in ARM_JOINTS]
        if any(value is None for value in actual):
            return
        for name, current, target in zip(ARM_JOINTS, actual, targets):
            lower, upper = JOINT_LIMITS[name]
            if (
                not math.isfinite(target)
                or target < lower
                or target > upper
                or abs(target - float(current)) > self.servo_max_joint_step
            ):
                self.get_logger().error(
                    "Rejected unsafe Servo target for %s" % name
                )
                return
        self._publish_command(ARM_JOINTS, targets)

    def _contact_callback(self, message):
        try:
            status = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if not isinstance(status, dict) or not status.get("contact_report_available", False):
            return
        if not all(key in status for key in ("left_finger_touching_table", "right_finger_touching_table")):
            return
        with self._state_lock:
            self._contact_time = time.monotonic()
            self._touching_table = bool(status["left_finger_touching_table"] or status["right_finger_touching_table"])
            if self._touching_table:
                self._trip_fault("Finger/table contact detected")

    def _trip_fault(self, reason):
        with self._state_lock:
            if not self._motion_fault:
                self._motion_fault = reason
                self._stop_generation += 1
                self._servo_stream_enabled = False
                self._fault_hold = dict(self._actual)
                self.get_logger().error("Motion stopped: " + reason)
            self._publish_command((), ())

    def _begin_scene_reset_callback(self, _request, response):
        """Give Isaac exclusive command ownership without enabling motion."""
        with self._state_lock:
            if any(lock.locked() for lock in self._controller_locks.values()):
                response.success = False
                response.message = "轨迹尚未停止，不能交接场景复位控制权"
                return response
            self._scene_reset_active = True
            self._motion_fault = self._motion_fault or "Scene reset in progress"
            self._stop_generation += 1
            self._servo_stream_enabled = False
            self._contact_time = 0.0
            self._last_state_time = 0.0
            self._joint_state_times.clear()
            response.success = True
            response.message = "场景复位已独占控制权，旧保持命令已暂停"
            return response

    def _clear_fault_callback(self, _request, response):
        with self._state_lock:
            now = time.monotonic()
            reasons = []
            for label, stamp in (("关节", self._last_state_time),
                                 ("接触", self._contact_time)):
                if stamp <= 0.0 or now - stamp > self.state_timeout:
                    reasons.append(f"{label}反馈未就绪或过期（年龄 {now - stamp:.3f}s，限值 {self.state_timeout:.3f}s）")
            if self._touching_table:
                reasons.append("夹指仍接触桌面")
            missing = [name for name in ALL_JOINTS if name not in self._actual]
            if missing:
                reasons.append(f"缺少关节：{missing}")
            active = [name for name, lock in self._controller_locks.items() if lock.locked()]
            if active:
                reasons.append(f"控制器尚未停止：{active}")
            if reasons:
                response.success = False
                response.message = "不能清除运动故障：" + "；".join(reasons)
                return response
            self._motion_fault = ""
            self._scene_reset_active = False
            self._fault_hold = {}
            self._path_error_since = None
            self._command = dict(self._actual)
            self._publish_command((), ())
            response.success = True
            response.message = "运动故障已清除；旧轨迹不会恢复"
            return response

    def _check_motion(self, joints):
        with self._state_lock:
            if self._motion_fault:
                return self._motion_fault
            if time.monotonic() - self._last_state_time > self.state_timeout:
                self._trip_fault("Joint feedback timeout")
            elif time.monotonic() - self._contact_time > self.state_timeout:
                self._trip_fault("Contact feedback timeout")
            elif tuple(joints) == ARM_JOINTS:
                # Compare against the command already sent, not a future waypoint.
                error = max(abs(self._command.get(n, self._actual[n]) - self._actual[n]) for n in joints)
                now = self._simulation_seconds()
                if error > self.path_error_limit:
                    if self._path_error_since is None:
                        self._path_error_since = now
                    if error > self.path_error_hard_limit or now - self._path_error_since >= self.path_error_duration:
                        self._trip_fault(f"Arm tracking error {error:.4f} rad")
                else:
                    self._path_error_since = None
            return self._motion_fault

    def _recording_status_callback(self, message):
        try:
            status = json.loads(message.data)
        except (ValueError, TypeError):
            return
        if not isinstance(status, dict):
            return
        with self._state_lock:
            self._recording = status.get("recording") is True
            self._recording_status_time = time.monotonic()

    def _motion_watchdog(self):
        with self._state_lock:
            active = self._servo_stream_enabled or any(
                lock.locked() for lock in self._controller_locks.values()
            )
            if active and not self._motion_fault and all(n in self._actual for n in ALL_JOINTS):
                self._check_motion(ARM_JOINTS)
            if (
                self._recording
                and time.monotonic() - self._recording_status_time <= self.state_timeout
                and not active
                and not self._scene_reset_active
                and not self._policy_control_active
                and all(n in self._command and n in self._actual for n in ALL_JOINTS)
            ):
                # Keep the last target alive between trajectories. Publishing
                # under the state lock serializes this with ownership changes;
                # _publish_command substitutes the measured fault hold if needed.
                self._check_motion(ARM_JOINTS)
                self._publish_command((), ())

    @staticmethod
    def _cancel_callback(_goal_handle):
        return CancelResponse.ACCEPT

    def _stop_motion_callback(self, _request, response):
        """Stop every controller loop and hold the measured joint positions."""
        with self._state_lock:
            self._stop_generation += 1
            hold_positions = [
                self._actual.get(name, self._command.get(name))
                for name in ALL_JOINTS
            ]
        if any(value is None for value in hold_positions):
            response.success = False
            response.message = "尚未收到完整关节状态，无法安全锁定当前位置"
            return response
        deadline = time.monotonic() + 2.0
        while (
            any(lock.locked() for lock in self._controller_locks.values())
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if any(lock.locked() for lock in self._controller_locks.values()):
            response.success = False
            response.message = "运动控制器未能在 2 秒内停止"
            return response
        self._publish_command(ALL_JOINTS, hold_positions)
        response.success = True
        response.message = "当前轨迹已停止，控制目标已锁定在实际关节位置"
        return response

    def _ordered_positions(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        expected_joints: Sequence[str],
    ) -> List[float]:
        values = dict(zip(names, positions))
        ordered = []
        for name in expected_joints:
            value = float(values[name])
            lower, upper = JOINT_LIMITS[name]
            ordered.append(max(lower, min(upper, value)))
        return ordered

    def _wait_for_state(self, joints: Iterable[str]) -> bool:
        deadline = time.monotonic() + self.state_timeout
        while time.monotonic() < deadline:
            with self._state_lock:
                if all(name in self._actual for name in joints):
                    return True
            time.sleep(0.02)
        return False

    def _simulation_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _publish_command(self, joints: Sequence[str], positions: Sequence[float]) -> None:
        with self._state_lock:
            if self._scene_reset_active:
                return
            if self._motion_fault:
                self._command = dict(self._fault_hold)
            else:
                self._command.update(zip(joints, positions))
            for name in ALL_JOINTS:
                if name not in self._command and name in self._actual:
                    self._command[name] = self._actual[name]
            names = [name for name in ALL_JOINTS if name in self._command]
            values = [self._command[name] for name in names]
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = names
            message.position = values
            # Serialize with fault latching: no old target can publish after hold.
            self._publisher.publish(message)

    def _feedback(
        self,
        goal_handle,
        joints: Sequence[str],
        desired: Sequence[float],
        elapsed: float,
    ) -> None:
        feedback = FollowJointTrajectory.Feedback()
        feedback.header.stamp = self.get_clock().now().to_msg()
        feedback.joint_names = list(joints)
        feedback.desired = JointTrajectoryPoint()
        feedback.desired.positions = list(desired)
        feedback.desired.time_from_start.sec = int(elapsed)
        feedback.desired.time_from_start.nanosec = int((elapsed % 1.0) * 1.0e9)
        with self._state_lock:
            actual = [self._actual.get(name, desired[index]) for index, name in enumerate(joints)]
        feedback.actual = JointTrajectoryPoint()
        feedback.actual.positions = actual
        feedback.error = JointTrajectoryPoint()
        feedback.error.positions = [target - value for target, value in zip(desired, actual)]
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _result(error_code: int, error_string: str):
        result = FollowJointTrajectory.Result()
        result.error_code = error_code
        result.error_string = error_string
        return result

    def _execute(self, goal_handle, joints: Sequence[str], controller: str):
        controller_lock = self._controller_locks[controller]
        if not controller_lock.acquire(blocking=False):
            goal_handle.abort()
            return self._result(
                FollowJointTrajectory.Result.INVALID_GOAL,
                f"{controller} controller is already executing a goal",
            )
        try:
            with self._state_lock:
                stop_generation = self._stop_generation
                servo_stream_enabled = self._servo_stream_enabled
                if controller == "arm":
                    self._path_error_since = None
            if controller == "arm" and servo_stream_enabled:
                goal_handle.abort()
                return self._result(
                    FollowJointTrajectory.Result.INVALID_GOAL,
                    "Servo stream is enabled; global arm trajectory rejected",
                )
            if not self._wait_for_state(joints):
                goal_handle.abort()
                return self._result(
                    FollowJointTrajectory.Result.INVALID_GOAL,
                    "No complete /joint_states received from Isaac Sim",
                )

            trajectory = goal_handle.request.trajectory
            point_targets = [
                self._ordered_positions(
                    trajectory.joint_names,
                    point.positions,
                    joints,
                )
                for point in trajectory.points
            ]
            point_times = [duration_seconds(point.time_from_start) for point in trajectory.points]
            with self._state_lock:
                segment_start = [self._actual[name] for name in joints]
            # MoveIt, robot_state_publisher and Isaac all use /clock. Using
            # monotonic wall time here makes the commanded trajectory diverge
            # whenever Isaac's real-time factor is not exactly 1.0.
            start_time = self._simulation_seconds()
            previous_point_time = 0.0
            period = 1.0 / max(1.0, self.command_rate)
            # Isaac applies /isaac_joint_commands once per simulation step.
            # Pacing the stream by wall-clock time outruns that consumer, so
            # the applied command keeps trailing the queued setpoint and the
            # tracking guard trips on pure latency.  Pace by simulation time
            # instead and never publish twice for the same simulation instant.
            wait_slice = min(period, 0.002)
            last_publish_sim = None

            for target, point_time in zip(point_targets, point_times):
                segment_duration = max(0.0, point_time - previous_point_time)
                while True:
                    fault = self._check_motion(joints)
                    if fault:
                        goal_handle.abort()
                        return self._result(FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED, fault)
                    with self._state_lock:
                        force_stopped = stop_generation != self._stop_generation
                    if force_stopped:
                        goal_handle.abort()
                        return self._result(
                            FollowJointTrajectory.Result.INVALID_GOAL,
                            "Trajectory stopped by episode reset",
                        )
                    if goal_handle.is_cancel_requested:
                        with self._state_lock:
                            hold = [self._actual[name] for name in joints]
                        self._publish_command(joints, hold)
                        goal_handle.canceled()
                        return self._result(
                            FollowJointTrajectory.Result.SUCCESSFUL,
                            "Trajectory canceled",
                        )
                    sim_now = self._simulation_seconds()
                    if (
                        last_publish_sim is not None
                        and sim_now - last_publish_sim < period
                    ):
                        time.sleep(wait_slice)
                        continue
                    last_publish_sim = sim_now
                    elapsed = sim_now - start_time
                    segment_elapsed = elapsed - previous_point_time
                    ratio = 1.0 if segment_duration == 0.0 else segment_elapsed / segment_duration
                    desired = interpolate(segment_start, target, ratio)
                    self._publish_command(joints, desired)
                    self._feedback(goal_handle, joints, desired, max(0.0, elapsed))
                    if ratio >= 1.0:
                        break
                segment_start = target
                previous_point_time = point_time

            deadline = self._simulation_seconds() + self.goal_time_tolerance
            within_tolerance_since = None
            required_stable_duration = (
                self.arm_goal_stable_duration if controller == "arm" else 0.0
            )
            while True:
                sim_now = self._simulation_seconds()
                if sim_now > deadline:
                    break
                fault = self._check_motion(joints)
                if fault:
                    goal_handle.abort()
                    return self._result(FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED, fault)
                with self._state_lock:
                    force_stopped = stop_generation != self._stop_generation
                if force_stopped:
                    goal_handle.abort()
                    return self._result(
                        FollowJointTrajectory.Result.INVALID_GOAL,
                        "Trajectory stopped by episode reset",
                    )
                if goal_handle.is_cancel_requested:
                    with self._state_lock:
                        hold = [self._actual[name] for name in joints]
                    self._publish_command(joints, hold)
                    goal_handle.canceled()
                    return self._result(
                        FollowJointTrajectory.Result.SUCCESSFUL,
                        "Trajectory canceled",
                    )
                self._publish_command(joints, point_targets[-1])
                with self._state_lock:
                    error = max(
                        abs(self._actual[name] - target)
                        for name, target in zip(joints, point_targets[-1])
                    )
                if error <= self.goal_tolerance:
                    if within_tolerance_since is None:
                        within_tolerance_since = sim_now
                    if (
                        sim_now - within_tolerance_since
                        >= required_stable_duration
                    ):
                        goal_handle.succeed()
                        return self._result(
                            FollowJointTrajectory.Result.SUCCESSFUL,
                            "Trajectory reached in Isaac Sim",
                        )
                else:
                    within_tolerance_since = None
                time.sleep(period)

            with self._state_lock:
                final_actual = [self._actual[name] for name in joints]
            errors = [
                abs(actual - target)
                for actual, target in zip(final_actual, point_targets[-1])
            ]
            worst_index = max(range(len(errors)), key=errors.__getitem__)
            # Keep the collision-checked endpoint active after a recoverable
            # miss. Retargeting to the measured position compounds the same
            # load-dependent tracking error during the next controller goal.
            self._publish_command(joints, point_targets[-1])
            goal_handle.abort()
            return self._result(
                FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                "Isaac Sim did not reach the final point within tolerance: "
                f"max_error={errors[worst_index]:.4f} rad, "
                f"joint={joints[worst_index]}, "
                f"controller_tolerance={self.goal_tolerance:.4f} rad",
            )
        except Exception as exc:
            self._trip_fault(f"Trajectory execution exception: {exc}")
            goal_handle.abort()
            return self._result(
                FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                self._motion_fault,
            )
        finally:
            controller_lock.release()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
