"""Expose ros2_control-compatible trajectory actions backed by Isaac JointState commands."""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, Iterable, List, Sequence

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
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
        self.declare_parameter("goal_tolerance", 0.005)
        self.declare_parameter("goal_time_tolerance", 3.0)
        self.declare_parameter("joint_limit_tolerance", 0.002)
        self.declare_parameter(
            "servo_command_topic", "/smolvla/servo_joint_trajectory"
        )
        self.declare_parameter("servo_max_joint_step_rad", 0.10)

        command_topic = self.get_parameter("command_topic").value
        state_topic = self.get_parameter("state_topic").value
        self.command_rate = float(self.get_parameter("command_rate").value)
        self.state_timeout = float(self.get_parameter("state_timeout").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.goal_time_tolerance = float(self.get_parameter("goal_time_tolerance").value)
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
        self._stop_generation = 0
        self._servo_stream_enabled = False

        self._publisher = self.create_publisher(JointState, command_topic, 10)
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
                    self._actual[name] = value
                    self._command.setdefault(name, value)
            if all(name in self._actual for name in ALL_JOINTS):
                self._last_state_time = time.monotonic()

    def _goal_callback(self, request, expected_joints: Sequence[str]):
        with self._state_lock:
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
            if self._controller_locks["arm"].locked():
                response.success = False
                response.message = "全局机械臂轨迹仍在执行，拒绝启用 Servo"
                return response
            with self._state_lock:
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

    def _servo_trajectory_callback(self, message: JointTrajectory) -> None:
        """Forward only explicitly enabled, small Servo joint increments."""
        with self._state_lock:
            enabled = self._servo_stream_enabled
            actual = [self._actual.get(name) for name in ARM_JOINTS]
        if not enabled or self._controller_locks["arm"].locked():
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

            for target, point_time in zip(point_targets, point_times):
                segment_duration = max(0.0, point_time - previous_point_time)
                while True:
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
                    elapsed = self._simulation_seconds() - start_time
                    segment_elapsed = elapsed - previous_point_time
                    ratio = 1.0 if segment_duration == 0.0 else segment_elapsed / segment_duration
                    desired = interpolate(segment_start, target, ratio)
                    self._publish_command(joints, desired)
                    self._feedback(goal_handle, joints, desired, max(0.0, elapsed))
                    if ratio >= 1.0:
                        break
                    time.sleep(period)
                segment_start = target
                previous_point_time = point_time

            deadline = self._simulation_seconds() + self.goal_time_tolerance
            while self._simulation_seconds() <= deadline:
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
                    goal_handle.succeed()
                    return self._result(
                        FollowJointTrajectory.Result.SUCCESSFUL,
                        "Trajectory reached in Isaac Sim",
                    )
                time.sleep(period)

            goal_handle.abort()
            return self._result(
                FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                "Isaac Sim did not reach the final point within tolerance",
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
