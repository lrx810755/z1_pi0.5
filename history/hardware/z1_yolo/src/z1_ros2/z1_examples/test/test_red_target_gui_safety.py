import math
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from moveit_msgs.msg import RobotTrajectory
from scipy.spatial.transform import Rotation
from trajectory_msgs.msg import JointTrajectoryPoint

from z1_examples.red_target_gui import (
    AUTO_SEQUENCE_NEXT_ACTION,
    ApprovedTrajectoryExecutor,
    DEFAULT_BOTTLE_Z_OFFSET_M,
    DEFAULT_BOTTLE_Y_OFFSET_M,
    DEFAULT_GRIPPER_CLOSE_DEG,
    DEFAULT_GRIPPER_OPEN_RAD,
    DEFAULT_GRASP_PITCH_DEG,
    DEFAULT_INSERTION_OFFSET_M,
    GRASP_PITCH_TOLERANCE_DEG,
    HOME_JOINT_TARGETS,
    HOME_POSITION_TOLERANCE_RAD,
    HOME_VELOCITY_TOLERANCE_RAD_S,
    add_gripper_motion,
    gripper_close_target_rad,
    horizontal_grasp_orientation,
    offset_along_base_y,
    offset_along_tool_x,
)
from z1_examples.red_target_handeye_approach import RedTargetHandeyeApproach


class GripperTrajectoryTest(unittest.TestCase):
    def test_gui_planning_inputs_do_not_require_a_second_rgb_image(self):
        planner = RedTargetHandeyeApproach.__new__(RedTargetHandeyeApproach)
        planner.args = SimpleNamespace(
            wait_timeout=0.1,
            base_frame="link00",
            gripper_frame="gripperStator",
        )
        planner.lock = threading.Lock()
        planner.latest_joint_positions = {
            f"joint{index}": 0.0 for index in range(1, 7)
        }
        planner.tf_buffer = Mock()
        planner.tf_buffer.can_transform.return_value = True

        with patch(
            "z1_examples.red_target_handeye_approach.rclpy.ok",
            return_value=True,
        ):
            planner.wait_for_planning_inputs()

        planner.tf_buffer.can_transform.assert_called_once()

    def test_insertion_advances_thirty_mm_along_pitched_tool_x(self):
        self.assertAlmostEqual(DEFAULT_INSERTION_OFFSET_M, 0.03)
        quat = horizontal_grasp_orientation([1.0, 0.0, 0.0], -5.0)
        start = np.array([0.40, 0.10, 0.20])
        target = offset_along_tool_x(start, quat, DEFAULT_INSERTION_OFFSET_M)
        displacement = target - start
        self.assertAlmostEqual(float(np.linalg.norm(displacement)), 0.03)
        self.assertGreater(displacement[0], 0.029)
        self.assertAlmostEqual(displacement[1], 0.0)
        self.assertAlmostEqual(displacement[2], -0.03 * math.sin(math.radians(5.0)))

    def test_horizontal_grasp_orientation_defaults_to_zero_pitch(self):
        quat = horizontal_grasp_orientation([1.0, 0.0, 0.4])
        rotation = Rotation.from_quat(quat).as_matrix()
        local_x_in_base = rotation[:, 0]
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0)
        self.assertAlmostEqual(
            float(np.rad2deg(np.arcsin(local_x_in_base[2]))), 0.0
        )
        self.assertGreater(local_x_in_base[0], 0.99)
        self.assertAlmostEqual(local_x_in_base[1], 0.0)
        self.assertAlmostEqual(DEFAULT_GRASP_PITCH_DEG, 0.0)
        self.assertAlmostEqual(GRASP_PITCH_TOLERANCE_DEG, 1.0)

    def test_horizontal_pose_constraint_keeps_joint6_at_ninety_degrees(self):
        planner = RedTargetHandeyeApproach.__new__(RedTargetHandeyeApproach)
        planner.args = SimpleNamespace(
            goal_tolerance=0.008,
            base_frame="link00",
            end_effector_link="gripper_tcp",
            max_joint_delta=0.12,
            local_joint_limit=0.0,
            target_joint6=math.pi / 2.0,
            target_joint6_tolerance=math.radians(2.0),
            orientation_weight=1.0,
            orientation_tolerance=math.radians(5.0),
            orientation_tolerances=(math.pi, math.radians(1.0), math.radians(5.0)),
            target_orientation_quat=horizontal_grasp_orientation([1.0, 0.0, 0.0]),
        )
        constraints = planner._position_constraints(np.array([0.5, 0.0, 0.2]))
        joint6 = next(
            item for item in constraints.joint_constraints if item.joint_name == "joint6"
        )
        orientation = constraints.orientation_constraints[0]
        self.assertAlmostEqual(joint6.position, math.pi / 2.0)
        self.assertAlmostEqual(joint6.tolerance_above, math.radians(2.0))
        self.assertAlmostEqual(orientation.absolute_x_axis_tolerance, math.pi)
        self.assertAlmostEqual(
            orientation.absolute_y_axis_tolerance, math.radians(1.0)
        )
        self.assertAlmostEqual(
            orientation.absolute_z_axis_tolerance, math.radians(5.0)
        )

    def test_named_open_target_matches_robot_grasp_backend(self):
        self.assertAlmostEqual(DEFAULT_GRIPPER_OPEN_RAD, -0.90)

    def test_bottle_approach_uses_lowered_offset(self):
        self.assertAlmostEqual(DEFAULT_BOTTLE_Z_OFFSET_M, -0.025)

    def test_final_pose_adds_twenty_mm_along_base_positive_y(self):
        self.assertAlmostEqual(DEFAULT_BOTTLE_Y_OFFSET_M, 0.02)
        start = np.array([0.40, -0.10, 0.20])
        target = offset_along_base_y(start, DEFAULT_BOTTLE_Y_OFFSET_M)
        np.testing.assert_allclose(target, [0.40, -0.08, 0.20])
        np.testing.assert_allclose(start, [0.40, -0.10, 0.20])

    def test_gripper_close_25_degrees_uses_z1_negative_joint_sign(self):
        self.assertAlmostEqual(DEFAULT_GRIPPER_CLOSE_DEG, 25.0)
        self.assertAlmostEqual(
            gripper_close_target_rad(25.0), -math.radians(25.0)
        )

    def test_home_targets_all_six_arm_joints_at_zero(self):
        self.assertEqual(
            HOME_JOINT_TARGETS,
            {f"joint{index}": 0.0 for index in range(1, 7)},
        )

    def test_home_has_local_relaxed_completion_tolerances(self):
        self.assertAlmostEqual(HOME_POSITION_TOLERANCE_RAD, 0.05)
        self.assertAlmostEqual(HOME_VELOCITY_TOLERANCE_RAD_S, 0.08)

    def test_home_completion_requires_consecutive_fresh_settled_states(self):
        executor = ApprovedTrajectoryExecutor.__new__(ApprovedTrajectoryExecutor)
        executor.lock = threading.Lock()
        executor.stop_requested = threading.Event()
        executor.joint_positions = dict(HOME_JOINT_TARGETS)
        executor.joint_velocities = {name: 0.01 for name in HOME_JOINT_TARGETS}
        executor.joint_state_generation = 0

        def publish_fresh_state(_node, timeout_sec):
            self.assertEqual(timeout_sec, 0.05)
            executor.joint_state_generation += 1

        with patch("z1_examples.red_target_gui.rclpy.ok", return_value=True), patch(
            "z1_examples.red_target_gui.rclpy.spin_once",
            side_effect=publish_fresh_state,
        ) as spin_once:
            worst_position, worst_velocity = executor.wait_for_settled_goal(
                HOME_JOINT_TARGETS,
                HOME_POSITION_TOLERANCE_RAD,
                HOME_VELOCITY_TOLERANCE_RAD_S,
                timeout=1.0,
                required_samples=3,
            )

        self.assertEqual(spin_once.call_count, 3)
        self.assertAlmostEqual(worst_position[1], 0.0)
        self.assertAlmostEqual(worst_velocity[1], 0.01)

    def test_auto_sequence_orders_approach_grasp_home_then_release(self):
        actions = []
        action = "approach"
        while action is not None:
            actions.append(action)
            action = AUTO_SEQUENCE_NEXT_ACTION.get(action)
        self.assertEqual(actions, ["approach", "grasp", "home", "release"])

    def test_gripper_motion_is_synchronised_and_field_lengths_match(self):
        robot_trajectory = RobotTrajectory()
        trajectory = robot_trajectory.joint_trajectory
        trajectory.joint_names = [f"joint{index}" for index in range(1, 7)]
        for seconds in (0, 1, 2):
            point = JointTrajectoryPoint()
            point.positions = [0.0] * 6
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6
            point.time_from_start.sec = seconds
            trajectory.points.append(point)

        add_gripper_motion(robot_trajectory, 0.0, -math.pi / 3.0, "jointGripper")

        self.assertEqual(trajectory.joint_names[-1], "jointGripper")
        self.assertAlmostEqual(trajectory.points[0].positions[-1], 0.0)
        self.assertAlmostEqual(trajectory.points[1].positions[-1], -math.pi / 6.0)
        self.assertAlmostEqual(trajectory.points[2].positions[-1], -math.pi / 3.0)
        for point in trajectory.points:
            self.assertEqual(len(point.positions), 7)
            self.assertEqual(len(point.velocities), 7)
            self.assertEqual(len(point.accelerations), 7)

    def test_rejects_duplicate_gripper_joint(self):
        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory.joint_names = ["jointGripper"]
        with self.assertRaisesRegex(RuntimeError, "已包含"):
            add_gripper_motion(robot_trajectory, 0.0, -1.0, "jointGripper")


if __name__ == "__main__":
    unittest.main()
