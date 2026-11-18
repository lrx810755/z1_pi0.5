#!/usr/bin/env python3

import argparse
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, MoveItErrorCodes, OrientationConstraint, PositionConstraint, RobotTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from z1_examples.target_detector import (
    TargetDetector,
    add_detector_arguments,
    detector_config_from_args,
)


def quaternion_to_matrix(x, y, z, w) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    xx = x * x * scale
    yy = y * y * scale
    zz = z * z * scale
    xy = x * y * scale
    xz = x * z * scale
    yz = y * z * scale
    wx = w * x * scale
    wy = w * y * scale
    wz = w * z * scale
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


class RedBallCameraMoveIt(Node):
    def __init__(self, args):
        super().__init__("red_ball_camera_moveit")
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.camera_matrix = None
        self.latest_color = None
        self.latest_depth = None
        self.latest_stamp = None
        self.latest_joint_state = None
        self.preview_failed = False
        self.camera_info_checked = False
        self.detector = TargetDetector(detector_config_from_args(args))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.move_group_client = ActionClient(self, MoveGroup, args.move_group_action)
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, args.execute_trajectory_action)
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, args.trajectory_action)

        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(Image, args.image_topic, self._color_cb, qos_profile_sensor_data)
        self.create_subscription(Image, args.depth_topic, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, args.joint_states_topic, self._joint_state_cb, 10)

    def _camera_info_cb(self, msg: CameraInfo):
        camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if not self.camera_info_checked:
            self._validate_camera_info(msg, camera_matrix)
            self.camera_info_checked = True
        self.camera_matrix = camera_matrix

    def _validate_camera_info(self, msg: CameraInfo, camera_matrix: np.ndarray):
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(f"Invalid CameraInfo focal length: fx={fx}, fy={fy}")

        projection = np.array(msg.p, dtype=np.float64).reshape(3, 4)
        expected_projection = np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        if not np.allclose(projection, expected_projection, atol=1e-6):
            self.get_logger().warning(
                "CameraInfo projection matrix does not match K. "
                "Use /camera/camera/depth/camera_info, not /camera/camera/depth/camera_info_raw."
            )

    def _color_cb(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_color = image
            self.latest_stamp = msg.header.stamp

    def _depth_cb(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = depth.astype(np.float32)
        with self.lock:
            self.latest_depth = depth

    def _joint_state_cb(self, msg: JointState):
        with self.lock:
            self.latest_joint_state = msg

    def wait_for_inputs(self):
        deadline = time.monotonic() + self.args.wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.lock:
                ready = self.camera_matrix is not None and self.latest_color is not None and self.latest_depth is not None
            if ready:
                return
        raise RuntimeError("Timed out waiting for color image, depth image, and CameraInfo")

    def detect_red_ball(self):
        with self.lock:
            if self.latest_color is None or self.latest_depth is None:
                raise RuntimeError("No synchronized camera data received yet")
            image = self.latest_color.copy()
            depth = self.latest_depth.copy()
            stamp = self.latest_stamp

        detection = self.detector.detect(image)
        if not detection.found:
            self._preview(image, mask=detection.mask)
            raise RuntimeError(f"Target not detected: {detection.reason}")
        u, v, area = detection.u, detection.v, detection.area

        depth_u, depth_v = self._color_to_depth_pixel(u, v)
        depth_m = self._depth_at(depth, depth_u, depth_v)
        camera_point = self._pixel_to_camera_point(depth_u, depth_v, depth_m)
        self._preview(
            image,
            mask=detection.mask,
            contour=detection.contour,
            target=(u, v, depth_m),
            bbox=detection.bbox if detection.contour is None else None,
            label=detection.label,
            confidence=detection.confidence,
        )
        return camera_point, (u, v), (depth_u, depth_v), area, depth_m, stamp

    def _color_to_depth_pixel(self, u, v):
        return (
            self.args.depth_u_scale * u + self.args.depth_u_offset,
            self.args.depth_v_scale * v + self.args.depth_v_offset,
        )

    def _depth_at(self, depth, u, v):
        x = int(round(u))
        y = int(round(v))
        if x < 0 or x >= depth.shape[1] or y < 0 or y >= depth.shape[0]:
            raise RuntimeError(
                f"Mapped depth pixel is outside the depth image: "
                f"u={u:.1f}, v={v:.1f}, image={depth.shape[1]}x{depth.shape[0]}. "
                "Move the red ball closer to the image center or adjust --depth-u/v-scale."
            )
        radius = int(self.args.depth_window_px)
        y0 = max(y - radius, 0)
        y1 = min(y + radius + 1, depth.shape[0])
        x0 = max(x - radius, 0)
        x1 = min(x + radius + 1, depth.shape[1])
        roi_depth = depth[y0:y1, x0:x1]
        valid = self._valid_depth(roi_depth)
        if valid.size == 0:
            raise RuntimeError("Red ball detected, but no valid depth near it")
        return float(np.median(valid))

    def _valid_depth(self, values):
        values = np.asarray(values).reshape(-1)
        values = values[np.isfinite(values)]
        return values[(values > self.args.min_depth_m) & (values < self.args.max_depth_m)]

    def _pixel_to_camera_point(self, u, v, depth_m):
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m, 1.0], dtype=np.float64)

    def lookup_transform_matrix(self, target_frame, source_frame, stamp=None):
        tf_time = Time.from_msg(stamp) if stamp is not None else Time()
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                tf_time,
                timeout=Duration(seconds=0.3),
            )
        except TransformException:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )
        return transform_to_matrix(transform)

    def compute_motion_target(self, camera_point, stamp):
        base_t_camera = self.lookup_transform_matrix(self.args.base_frame, self.args.camera_frame, stamp)
        base_t_ee = self.lookup_transform_matrix(self.args.base_frame, self.args.end_effector_link, stamp)
        target_base_h = base_t_camera @ camera_point
        target_base = target_base_h[:3]
        ee_base = base_t_ee[:3, 3]
        direction = target_base - ee_base
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            raise RuntimeError("Red ball is too close to end effector origin")
        direction /= norm
        if self.args.step_size > 0.0:
            motion_target = ee_base + direction * min(self.args.step_size, norm)
        else:
            motion_target = target_base - direction * self.args.approach_distance
        motion_target[2] += self.args.z_offset
        return target_base, ee_base, motion_target

    def plan_to_position(self, position):
        if not self.move_group_client.wait_for_server(timeout_sec=self.args.action_wait_timeout):
            raise RuntimeError(f"MoveGroup action unavailable: {self.args.move_group_action}")

        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = self.args.group
        request.num_planning_attempts = self.args.planning_attempts
        request.allowed_planning_time = self.args.planning_time
        request.max_velocity_scaling_factor = self.args.speed_scale
        request.max_acceleration_scaling_factor = self.args.speed_scale
        request.start_state.is_diff = True
        request.goal_constraints = [self._goal_constraints(position)]

        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("MoveGroup rejected the goal")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveGroup failed with error_code={result.error_code.val}")
        return result.planned_trajectory.joint_trajectory

    def plan_safe_step(self, start_position, requested_position):
        step = requested_position - start_position
        last_error = None
        fallback = None
        scale = 1.0
        for _ in range(self.args.max_step_halving + 1):
            candidate = start_position + step * scale
            try:
                trajectory = self.plan_to_position(candidate)
            except RuntimeError as exc:
                last_error = exc
                scale *= 0.5
                continue
            max_delta = self._max_joint_delta(trajectory)
            if fallback is None or max_delta < fallback[2]:
                fallback = (trajectory, candidate, max_delta, scale)
            if max_delta <= self.args.max_joint_delta:
                return trajectory, candidate, max_delta, scale
            last_error = RuntimeError(
                f"planned joint delta {max_delta:.3f} rad exceeds "
                f"--max-joint-delta {self.args.max_joint_delta:.3f} rad"
            )
            scale *= 0.5
            if np.linalg.norm(step * scale) < self.args.min_step_size:
                break
        if fallback is not None and not self.args.execute:
            return fallback
        raise RuntimeError(f"Unable to find a safe step after shrinking target: {last_error}")

    def execute_trajectory(self, trajectory):
        if self.args.execution_mode == "moveit":
            self.execute_trajectory_moveit(trajectory)
        else:
            self.execute_trajectory_controller(trajectory)

    def execute_trajectory_moveit(self, trajectory):
        if not trajectory.joint_names or not trajectory.points:
            raise RuntimeError("MoveGroup returned an empty trajectory")
        max_delta = self._max_joint_delta(trajectory)
        if max_delta > self.args.max_joint_delta:
            raise RuntimeError(
                f"Refusing to execute: planned joint delta {max_delta:.3f} rad exceeds "
                f"--max-joint-delta {self.args.max_joint_delta:.3f} rad"
            )
        if not self.execute_trajectory_client.wait_for_server(timeout_sec=self.args.action_wait_timeout):
            raise RuntimeError(f"MoveIt execute trajectory action unavailable: {self.args.execute_trajectory_action}")

        start_positions = self._joint_positions(trajectory.joint_names)
        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory = trajectory
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_trajectory

        future = self.execute_trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("MoveIt rejected trajectory execution")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt trajectory execution failed: error_code={result.error_code.val}")
        try:
            observed_delta, final_error = self._verify_execution(trajectory, start_positions)
        except RuntimeError as exc:
            if not self.args.final_correction:
                raise
            print(f"MoveIt execution verification failed: {exc}")
            print("Sending final joint correction through joint_trajectory_controller")
            correction = self._final_correction_trajectory(trajectory)
            self.execute_trajectory_controller(correction, check_max_delta=False)
            return
        print(f"Execution observed joint motion: {observed_delta:.3f} rad")
        print(f"Execution final joint error:    {final_error:.3f} rad")

    def execute_trajectory_controller(self, trajectory, check_max_delta=True):
        if not trajectory.joint_names or not trajectory.points:
            raise RuntimeError("MoveGroup returned an empty trajectory")
        max_delta = self._max_joint_delta(trajectory)
        if check_max_delta and max_delta > self.args.max_joint_delta:
            raise RuntimeError(
                f"Refusing to execute: planned joint delta {max_delta:.3f} rad exceeds "
                f"--max-joint-delta {self.args.max_joint_delta:.3f} rad"
            )
        if not self.trajectory_client.wait_for_server(timeout_sec=self.args.action_wait_timeout):
            raise RuntimeError(f"Trajectory action unavailable: {self.args.trajectory_action}")
        start_positions = self._joint_positions(trajectory.joint_names)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Joint trajectory controller rejected goal")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f"Trajectory execution failed: {result.error_code} {result.error_string}")
        observed_delta, final_error = self._verify_execution(trajectory, start_positions)
        print(f"Execution observed joint motion: {observed_delta:.3f} rad")
        print(f"Execution final joint error:    {final_error:.3f} rad")

    def _final_correction_trajectory(self, trajectory):
        correction = JointTrajectory()
        correction.joint_names = list(trajectory.joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(trajectory.points[-1].positions)
        if trajectory.points[-1].velocities:
            point.velocities = [0.0] * len(correction.joint_names)
        duration = Duration(seconds=self.args.final_correction_duration).to_msg()
        point.time_from_start = duration
        correction.points.append(point)
        return correction

    def _joint_positions(self, joint_names):
        deadline = time.monotonic() + self.args.action_wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.lock:
                msg = self.latest_joint_state
            if msg is None:
                continue
            positions = dict(zip(msg.name, msg.position))
            if all(name in positions for name in joint_names):
                return np.array([positions[name] for name in joint_names], dtype=np.float64)
        raise RuntimeError(f"Timed out waiting for joint states: {joint_names}")

    def _verify_execution(self, trajectory, start_positions):
        target_positions = np.array(trajectory.points[-1].positions, dtype=np.float64)
        deadline = time.monotonic() + self.args.execution_verify_timeout
        best_final_error = float("inf")
        best_observed_delta = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            current_positions = self._joint_positions(trajectory.joint_names)
            observed_delta = float(np.max(np.abs(current_positions - start_positions)))
            final_error = float(np.max(np.abs(target_positions - current_positions)))
            best_observed_delta = max(best_observed_delta, observed_delta)
            best_final_error = min(best_final_error, final_error)
            if final_error <= self.args.execution_joint_tolerance:
                return observed_delta, final_error
            time.sleep(0.05)
        current_positions = self._joint_positions(trajectory.joint_names)
        target_positions = np.array(trajectory.points[-1].positions, dtype=np.float64)
        per_joint = ", ".join(
            f"{name}: target={target:.3f}, current={current:.3f}, err={abs(target-current):.3f}"
            for name, target, current in zip(trajectory.joint_names, target_positions, current_positions)
        )
        raise RuntimeError(
            f"Trajectory action reported success, but joint states did not reach the target: "
            f"best final error {best_final_error:.3f} rad, observed motion {best_observed_delta:.3f} rad; "
            f"{per_joint}"
        )

    def _goal_constraints(self, position):
        constraints = Constraints()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [self.args.goal_tolerance]
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.w = 1.0
        constraint = PositionConstraint()
        constraint.header.frame_id = self.args.base_frame
        constraint.link_name = self.args.end_effector_link
        constraint.constraint_region.primitives.append(primitive)
        constraint.constraint_region.primitive_poses.append(pose)
        constraint.weight = 1.0
        constraints.position_constraints.append(constraint)

        current_ee = self.tf_buffer.lookup_transform(
            self.args.base_frame,
            self.args.end_effector_link,
            Time(),
            timeout=Duration(seconds=0.5),
        )
        orientation = OrientationConstraint()
        orientation.header.frame_id = self.args.base_frame
        orientation.link_name = self.args.end_effector_link
        orientation.orientation = current_ee.transform.rotation
        orientation.absolute_x_axis_tolerance = self.args.orientation_tolerance
        orientation.absolute_y_axis_tolerance = self.args.orientation_tolerance
        orientation.absolute_z_axis_tolerance = self.args.orientation_tolerance
        orientation.weight = 1.0
        constraints.orientation_constraints.append(orientation)
        return constraints

    def _max_joint_delta(self, trajectory):
        if len(trajectory.points) < 2:
            return 0.0
        start = np.array(trajectory.points[0].positions, dtype=np.float64)
        end = np.array(trajectory.points[-1].positions, dtype=np.float64)
        return float(np.max(np.abs(end - start)))

    def _trajectory_duration(self, trajectory):
        if not trajectory.points:
            return 0.0
        stamp = trajectory.points[-1].time_from_start
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _preview(self, image, mask=None, contour=None, target=None, bbox=None, label=None, confidence=None, red_ratio=None):
        if self.args.no_preview or self.preview_failed:
            return
        try:
            overlay = image.copy()
            if contour is not None:
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
            if bbox is not None:
                x1, y1, x2, y2 = [int(round(value)) for value in bbox]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = str(label) if label is not None else "yolo"
                if confidence is not None:
                    text += f" {confidence:.2f}"
                if red_ratio is not None:
                    text += f" red={red_ratio:.2f}"
                cv2.putText(overlay, text, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if target is not None:
                u, v, depth_m = target
                cv2.circle(overlay, (int(round(u)), int(round(v))), 5, (255, 255, 255), -1)
                cv2.putText(overlay, f"red ball z={depth_m:.3f}m", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("red ball camera moveit", overlay)
            if mask is not None:
                cv2.imshow("red ball mask", mask)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.preview_failed = True
            self.get_logger().warning(f"Preview disabled: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Detect a red ball from the wrist camera and plan a Z1 MoveIt approach motion.")
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/depth/camera_info")
    parser.add_argument("--camera-frame", default="camera_depth_optical_frame")
    parser.add_argument("--base-frame", default="link00")
    parser.add_argument("--end-effector-link", default="link06")
    parser.add_argument("--group", default="z1_arm")
    parser.add_argument("--move-group-action", default="/move_action")
    parser.add_argument("--execute-trajectory-action", default="/execute_trajectory")
    parser.add_argument("--trajectory-action", default="/joint_trajectory_controller/follow_joint_trajectory")
    parser.add_argument("--execution-mode", choices=["moveit", "controller"], default="moveit")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--step-size", type=float, default=0.0)
    parser.add_argument("--min-step-size", type=float, default=0.004)
    parser.add_argument("--max-step-halving", type=int, default=4)
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.02)
    parser.add_argument("--orientation-tolerance", type=float, default=0.25)
    parser.add_argument("--speed-scale", type=float, default=0.05)
    parser.add_argument("--max-joint-delta", type=float, default=1.2)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--planning-attempts", type=int, default=5)
    parser.add_argument("--action-wait-timeout", type=float, default=10.0)
    parser.add_argument("--execution-verify-timeout", type=float, default=8.0)
    parser.add_argument("--execution-joint-tolerance", type=float, default=0.03)
    parser.add_argument("--final-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-correction-duration", type=float, default=5.0)
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    parser.add_argument("--detect-timeout", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--stop-distance", type=float, default=0.12)
    parser.add_argument("--settle-time", type=float, default=0.25)
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-preview", action="store_true")
    add_detector_arguments(parser)
    parser.add_argument("--min-area-px", type=float, default=150.0)
    parser.add_argument("--min-saturation", type=int, default=80)
    parser.add_argument("--min-value", type=int, default=50)
    parser.add_argument("--hue-high1", type=int, default=10)
    parser.add_argument("--hue-low2", type=int, default=170)
    parser.add_argument("--depth-window-px", type=int, default=20)
    parser.add_argument("--depth-u-scale", type=float, default=2.0)
    parser.add_argument("--depth-v-scale", type=float, default=2.0)
    parser.add_argument("--depth-u-offset", type=float, default=0.0)
    parser.add_argument("--depth-v-offset", type=float, default=0.0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    args.speed_scale = max(min(args.speed_scale, 1.0), 0.01)
    return args


def main():
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = RedBallCameraMoveIt(args)
    try:
        node.wait_for_inputs()
        iterations = max(args.iterations if args.execute else 1, 1)
        for iteration in range(1, iterations + 1):
            deadline = time.monotonic() + args.detect_timeout
            last_error = None
            result = None
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                try:
                    result = node.detect_red_ball()
                    break
                except RuntimeError as exc:
                    last_error = exc
            if result is None:
                raise RuntimeError(f"Timed out detecting red ball: {last_error}")

            camera_point, pixel, depth_pixel, area, depth_m, stamp = result
            target_base, ee_base, motion_target = node.compute_motion_target(camera_point, stamp)
            distance = float(np.linalg.norm(target_base - ee_base))
            print(f"Red ball detection [{iteration}/{iterations}]")
            print(f"  pixel:      u={pixel[0]:.1f}, v={pixel[1]:.1f}, area={area:.1f}px")
            print(f"  depth px:   u={depth_pixel[0]:.1f}, v={depth_pixel[1]:.1f}")
            print(f"  depth:      {depth_m:.4f} m")
            print(f"  distance:   {distance:.4f} m")
            print(f"  target:     x={target_base[0]:.4f}, y={target_base[1]:.4f}, z={target_base[2]:.4f} in {args.base_frame}")
            print(f"  current ee: x={ee_base[0]:.4f}, y={ee_base[1]:.4f}, z={ee_base[2]:.4f} in {args.base_frame}")

            if distance <= args.stop_distance:
                print(f"Stop: end effector is within {args.stop_distance:.3f} m of the red ball")
                break

            print(f"  command ee: x={motion_target[0]:.4f}, y={motion_target[1]:.4f}, z={motion_target[2]:.4f} in {args.base_frame}")
            trajectory, selected_target, max_delta, step_scale = node.plan_safe_step(ee_base, motion_target)
            if step_scale < 1.0:
                print(
                    f"  safe step:  scaled to {step_scale:.3f}, "
                    f"x={selected_target[0]:.4f}, y={selected_target[1]:.4f}, z={selected_target[2]:.4f}"
                )
            print(f"  joint max:  {max_delta:.3f} rad")
            print(f"MoveIt plan: {len(trajectory.points)} trajectory points")
            print(f"MoveIt duration: {node._trajectory_duration(trajectory):.3f} s")
            if args.execute:
                node.execute_trajectory(trajectory)
                print("Execution: success")
                time.sleep(args.settle_time)
            else:
                print("Execution: skipped; add --execute to move")
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
