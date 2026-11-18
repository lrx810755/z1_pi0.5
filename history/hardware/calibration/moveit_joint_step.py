#!/usr/bin/env python3

import argparse
import sys
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, DisplayTrajectory, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState


ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


class SafeJointStep(Node):
    def __init__(self):
        super().__init__("safe_handeye_joint_step")
        self.positions = {}
        self.velocities = {}
        self.joint_state_time = 0.0
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.move_group = ActionClient(self, MoveGroup, "/move_action")
        self.trajectory = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.display = self.create_publisher(DisplayTrajectory, "/display_planned_path", 10)

    def _joint_state_cb(self, msg):
        self.positions.update(zip(msg.name, msg.position))
        if len(msg.velocity) == len(msg.name):
            self.velocities.update(zip(msg.name, msg.velocity))
        self.joint_state_time = time.monotonic()

    def wait_for_joint_state(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(name in self.positions for name in ARM_JOINTS):
                return
        raise RuntimeError("Timed out waiting for all arm joints")

    def current(self):
        return np.array([self.positions[name] for name in ARM_JOINTS], dtype=float)

    def plan(self, target, speed):
        if not self.move_group.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("MoveGroup action server is unavailable")

        constraints = Constraints(name="handeye_safe_joint_step")
        for name, value in zip(ARM_JOINTS, target):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(value)
            constraint.tolerance_above = 0.005
            constraint.tolerance_below = 0.005
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)

        goal = MoveGroup.Goal()
        goal.request.group_name = "z1_arm"
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = speed
        goal.request.max_acceleration_scaling_factor = speed
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("MoveGroup rejected the planning goal")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("MoveGroup returned no result")
        result = wrapped.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveGroup planning failed: {result.error_code.val}")

        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory.append(result.planned_trajectory)
        self.display.publish(display)
        return result.planned_trajectory.joint_trajectory

    def validate(self, trajectory, max_step):
        if not trajectory.joint_names or not trajectory.points:
            raise RuntimeError("Planner returned an empty trajectory")
        positions = np.asarray([point.positions for point in trajectory.points], dtype=float)
        peak = np.max(np.abs(positions - positions[0]), axis=0)
        max_delta = float(np.max(peak))
        if max_delta > max_step:
            details = ", ".join(
                f"{name}={delta:.4f}" for name, delta in zip(trajectory.joint_names, peak)
            )
            raise RuntimeError(
                f"Planned motion {max_delta:.4f} rad exceeds {max_step:.4f} rad: {details}"
            )
        return max_delta, dict(zip(trajectory.joint_names, peak))

    def execute(self, trajectory):
        if not self.trajectory.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Joint trajectory controller action is unavailable")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        future = self.trajectory.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("Controller rejected the trajectory")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("Controller returned no result")
        if wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Execution failed: {wrapped.result.error_code} {wrapped.result.error_string}"
            )

    def wait_for_settle(self, trajectory, timeout=10.0):
        target = dict(zip(trajectory.joint_names, trajectory.points[-1].positions))
        deadline = time.monotonic() + timeout
        history = []
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if not all(name in self.positions for name in ARM_JOINTS):
                continue
            now = time.monotonic()
            current = np.array([self.positions[name] for name in ARM_JOINTS], dtype=float)
            history.append((now, current))
            history = [(stamp, values) for stamp, values in history if now - stamp <= 0.75]
            error = max(abs(self.positions[name] - target[name]) for name in ARM_JOINTS)
            if len(history) >= 2 and history[-1][0] - history[0][0] >= 0.6:
                positions = np.array([values for _, values in history])
                position_span = float(np.max(np.ptp(positions, axis=0)))
                if error <= 0.02 and position_span <= 0.006:
                    print(
                        f"SETTLED max_error={error:.4f} rad "
                        f"position_span={position_span:.4f} rad"
                    )
                    return
        raise RuntimeError("Robot did not settle within the safety timeout")


def parse_delta(values):
    result = {name: 0.0 for name in ARM_JOINTS}
    for item in values:
        name, value = item.split("=", 1)
        if name not in result:
            raise ValueError(f"Unknown arm joint: {name}")
        result[name] = float(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", action="append", default=[])
    parser.add_argument("--speed", type=float, default=0.03)
    parser.add_argument("--max-step", type=float, default=0.12)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    delta = parse_delta(args.delta)

    rclpy.init()
    node = SafeJointStep()
    try:
        node.wait_for_joint_state()
        start = node.current()
        target = start + np.array([delta[name] for name in ARM_JOINTS], dtype=float)
        print("start:  " + " ".join(f"{name}={value:+.4f}" for name, value in zip(ARM_JOINTS, start)))
        print("target: " + " ".join(f"{name}={value:+.4f}" for name, value in zip(ARM_JOINTS, target)))
        trajectory = node.plan(target, max(0.01, min(args.speed, 0.1)))
        max_delta, per_joint = node.validate(trajectory, args.max_step)
        print(f"PLAN_OK points={len(trajectory.points)} max_delta={max_delta:.4f} rad")
        print("trajectory: " + " ".join(f"{name}={value:.4f}" for name, value in per_joint.items()))
        if not args.execute:
            if input("Type EXECUTE to move the real robot: ").strip() != "EXECUTE":
                print("Execution cancelled")
                return 2
        node.execute(trajectory)
        print("EXECUTION_OK")
        node.wait_for_settle(trajectory)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
