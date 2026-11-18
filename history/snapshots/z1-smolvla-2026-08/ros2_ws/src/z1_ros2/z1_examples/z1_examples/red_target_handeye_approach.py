#!/usr/bin/env python3

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, JointState
from scipy.spatial.transform import Rotation
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from z1_examples.target_detector import (
    TargetDetector,
    add_detector_arguments,
    detector_config_from_args,
)


DEFAULT_CALIBRATION_PATH = Path.home() / "ros2_ws" / "z1_handeye_charuco_gripper_to_camera.yaml"


def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def load_handeye_matrix(path: Path) -> np.ndarray:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "matrix" in data:
        return np.array(data["matrix"], dtype=np.float64)

    translation = data["translation"]
    rotation = data["rotation_xyzw"]
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([
        rotation["x"],
        rotation["y"],
        rotation["z"],
        rotation["w"],
    ]).as_matrix()
    matrix[:3, 3] = [translation["x"], translation["y"], translation["z"]]
    return matrix


class RedTargetHandeyeApproach(Node):
    def __init__(self, args):
        super().__init__("red_target_handeye_approach")
        self.args = args
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.latest_image = None
        self.latest_depth = None
        self.latest_image_stamp = None
        self.latest_joint_positions = {}
        self.last_missing_report = 0.0
        self.lock = threading.Lock()
        self.preview_failed = False
        self.detector = TargetDetector(detector_config_from_args(args))
        self.gripper_t_camera = load_handeye_matrix(Path(args.calibration))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.move_group_client = ActionClient(self, MoveGroup, args.move_group_action)
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, args.trajectory_action)
        self.display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )
        self.start_event = threading.Event()
        self.start_service = None
        if args.wait_for_trigger:
            self.start_service = self.create_service(
                Trigger, args.trigger_service, self._start_callback
            )

        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(Image, args.image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        if args.depth_topic:
            self.create_subscription(Image, args.depth_topic, self._depth_cb, qos_profile_sensor_data)

    def _start_callback(self, request, response):
        del request
        if self.start_event.is_set():
            response.success = False
            response.message = "Red target detection has already been triggered"
            return response
        self.start_event.set()
        response.success = True
        response.message = "Starting red target detection and planning"
        return response

    def wait_for_start_trigger(self):
        if not self.args.wait_for_trigger:
            return
        self.get_logger().info(
            f"Ready. Call {self.args.trigger_service} to start red target detection"
        )
        while rclpy.ok() and not self.start_event.is_set():
            rclpy.spin_once(self, timeout_sec=0.1)
        if not rclpy.ok():
            raise KeyboardInterrupt

    def _camera_info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _image_cb(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = msg.header.stamp

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
            self.latest_joint_positions = {
                name: float(position) for name, position in zip(msg.name, msg.position)
            }

    def wait_for_inputs(self):
        deadline = time.monotonic() + self.args.wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            with self.lock:
                has_image = self.latest_image is not None
                has_depth = self.latest_depth is not None or self.args.target_distance > 0.0
                has_joint_state = all(
                    f"joint{index}" in self.latest_joint_positions for index in range(1, 7)
                )
            missing = []
            if self.camera_matrix is None:
                missing.append("CameraInfo")
            if not has_image:
                missing.append("RGB image")
            if not has_depth:
                missing.append("depth image or --target-distance")
            if not has_joint_state:
                missing.append("JointState")
            if self.camera_matrix is not None and has_image and has_depth and has_joint_state:
                return
            now = time.monotonic()
            if now - self.last_missing_report > 1.0:
                self.get_logger().info("Waiting for " + ", ".join(missing))
                self.last_missing_report = now
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("Timed out waiting for image, CameraInfo, depth, and JointState")

    def detect_target(self, raise_on_missing=True):
        with self.lock:
            if self.latest_image is None:
                raise RuntimeError("No image received")
            image = self.latest_image.copy()
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            stamp = self.latest_image_stamp

        detection = self.detector.detect(image)
        if not detection.found:
            self._show_preview(image, detection, None)
            if not raise_on_missing:
                return None
            raise RuntimeError(f"Target not found: {detection.reason}")

        u, v, area = detection.u, detection.v, detection.area
        depth_m = self._target_depth(depth, detection, u, v)
        camera_point = self._pixel_to_camera_point(u, v, depth_m)
        self._show_preview(image, detection, (u, v, depth_m, camera_point))
        return camera_point, (u, v), area, depth_m, stamp

    def wait_for_target(self):
        deadline = time.monotonic() + self.args.detect_timeout
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                result = self.detect_target(raise_on_missing=False)
                if result is not None:
                    return result
            except RuntimeError as exc:
                last_error = exc
                self.get_logger().warning(str(exc), throttle_duration_sec=1.0)
            time.sleep(0.05)
        if last_error is not None:
            raise RuntimeError(f"Timed out detecting red target: {last_error}")
        raise RuntimeError("Timed out detecting red target")

    def _target_depth(self, depth, detection, u, v):
        if depth is None:
            return float(self.args.target_distance)

        x = int(round(u))
        y = int(round(v))
        radius = int(self.args.depth_window_px)
        y0 = max(y - radius, 0)
        y1 = min(y + radius + 1, depth.shape[0])
        x0 = max(x - radius, 0)
        x1 = min(x + radius + 1, depth.shape[1])
        roi_depth = depth[y0:y1, x0:x1]
        if detection.mask is not None:
            roi_mask = detection.mask[y0:y1, x0:x1] > 0
        elif detection.bbox is not None:
            roi_mask = np.ones(roi_depth.shape, dtype=bool)
        else:
            roi_mask = np.ones(roi_depth.shape, dtype=bool)
        valid = self._valid_depth_values(roi_depth[roi_mask])
        if valid.size == 0:
            valid = self._valid_depth_values(roi_depth)
        if valid.size == 0 and self.args.target_distance > 0.0:
            self.get_logger().warning(
                f"No valid depth near red target; using fallback distance {self.args.target_distance:.3f} m",
                throttle_duration_sec=1.0,
            )
            return float(self.args.target_distance)
        if valid.size == 0:
            raise RuntimeError("Red target found, but no valid depth at the target")
        return float(np.median(valid))

    def _valid_depth_values(self, depth_values):
        valid = np.asarray(depth_values).reshape(-1)
        valid = valid[np.isfinite(valid)]
        return valid[(valid > self.args.min_depth_m) & (valid < self.args.max_depth_m)]

    def _pixel_to_camera_point(self, u, v, depth_m):
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        return np.array([
            (u - cx) * depth_m / fx,
            (v - cy) * depth_m / fy,
            depth_m,
            1.0,
        ], dtype=np.float64)

    def lookup_base_t_gripper(self, stamp):
        tf_time = Time.from_msg(stamp) if stamp is not None else Time()
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.base_frame,
                self.args.gripper_frame,
                tf_time,
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            if not self.args.allow_latest_tf_fallback:
                raise RuntimeError(
                    f"Cannot lookup TF {self.args.base_frame} <- {self.args.gripper_frame} at image time: {exc}"
                ) from exc
            self.get_logger().warning(
                "Image-time TF lookup failed; falling back to latest TF. "
                "Keep the robot still for this test. Original error: " + str(exc)
            )
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.base_frame,
                    self.args.gripper_frame,
                    Time(),
                    timeout=Duration(seconds=0.5),
                )
            except TransformException as latest_exc:
                raise RuntimeError(
                    f"Cannot lookup latest TF {self.args.base_frame} <- {self.args.gripper_frame}: {latest_exc}"
                ) from latest_exc
        return transform_to_matrix(transform)

    def compute_approach_point(self, camera_point, stamp):
        base_t_gripper = self.lookup_base_t_gripper(stamp)
        base_t_camera = base_t_gripper @ self.gripper_t_camera
        target_base = base_t_camera @ camera_point
        camera_origin_base = base_t_camera @ np.array([0.0, 0.0, 0.0, 1.0])
        approach_dir = target_base[:3] - camera_origin_base[:3]
        norm = float(np.linalg.norm(approach_dir))
        if norm < 1e-6:
            raise RuntimeError("Target is too close to camera origin")
        approach_dir /= norm
        approach_base = target_base[:3] - approach_dir * self.args.approach_distance
        approach_base[2] += self.args.z_offset
        gripper_base = base_t_gripper[:3, 3]

        motion_target = approach_base
        motion_mode = "absolute approach point"
        if self.args.step_size > 0.0:
            step_dir = target_base[:3] - gripper_base
            step_norm = float(np.linalg.norm(step_dir))
            if step_norm < 1e-6:
                raise RuntimeError("Target is too close to gripper origin")
            step_dir /= step_norm
            motion_target = gripper_base + step_dir * min(self.args.step_size, step_norm)
            motion_target[2] += self.args.z_offset
            motion_mode = f"incremental step {self.args.step_size:.3f} m"

        return target_base[:3], approach_base, motion_target, camera_origin_base[:3], gripper_base, motion_mode

    def compute_image_servo_point(self, pixel, stamp):
        if self.camera_matrix is None:
            raise RuntimeError("No CameraInfo received")

        base_t_gripper = self.lookup_base_t_gripper(stamp)
        base_t_camera = base_t_gripper @ self.gripper_t_camera
        gripper_base = base_t_gripper[:3, 3]
        camera_origin_base = base_t_camera @ np.array([0.0, 0.0, 0.0, 1.0])

        u, v = pixel
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])
        err_u = float(u - cx)
        err_v = float(v - cy)

        delta_x = np.clip((err_u / fx) * self.args.servo_gain, -self.args.servo_step_size, self.args.servo_step_size)
        delta_y = np.clip((err_v / fy) * self.args.servo_gain, -self.args.servo_step_size, self.args.servo_step_size)
        centered = abs(err_u) <= self.args.servo_deadband_px and abs(err_v) <= self.args.servo_deadband_px
        delta_z = self.args.servo_forward_step if centered else 0.0

        camera_delta = np.array([delta_x, delta_y, delta_z], dtype=np.float64)
        base_delta = base_t_camera[:3, :3] @ camera_delta
        motion_target = gripper_base + base_delta
        motion_target[2] += self.args.z_offset
        motion_mode = (
            f"image servo du={err_u:.1f}px dv={err_v:.1f}px "
            f"camera_delta=({camera_delta[0]:.3f},{camera_delta[1]:.3f},{camera_delta[2]:.3f}) m"
        )
        return motion_target, camera_origin_base[:3], gripper_base, motion_mode, centered

    def plan_move_group_goal(self, position):
        self.get_logger().info(f"Waiting for MoveGroup action server: {self.args.move_group_action}")
        if not self.move_group_client.wait_for_server(timeout_sec=self.args.action_wait_timeout):
            raise RuntimeError(f"MoveGroup action server unavailable: {self.args.move_group_action}")

        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = self.args.group
        request.num_planning_attempts = self.args.planning_attempts
        request.allowed_planning_time = self.args.planning_time
        request.max_velocity_scaling_factor = self.args.speed_scale
        request.max_acceleration_scaling_factor = self.args.speed_scale
        request.start_state.is_diff = True
        request.goal_constraints = [self._position_constraints(position)]

        options = goal.planning_options
        options.plan_only = True
        options.look_around = False
        options.replan = False
        options.planning_scene_diff.is_diff = True
        options.planning_scene_diff.robot_state.is_diff = True

        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("MoveGroup rejected the goal")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            raise RuntimeError("MoveGroup returned no result")
        code = result.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveGroup failed, error_code={code}")
        return result.result

    def publish_display_trajectory(self, result):
        display = DisplayTrajectory()
        display.trajectory_start = result.trajectory_start
        display.trajectory.append(result.planned_trajectory)
        self.display_trajectory_publisher.publish(display)
        self.get_logger().info("Published planned path to /display_planned_path")

    def trajectory_joint_delta(self, trajectory):
        if not trajectory.joint_names or len(trajectory.points) < 2:
            return 0.0, {}
        start = np.array(trajectory.points[0].positions, dtype=np.float64)
        positions = np.array([point.positions for point in trajectory.points], dtype=np.float64)
        deviations = positions - start
        peak_indices = np.argmax(np.abs(deviations), axis=0)
        deltas = deviations[peak_indices, np.arange(deviations.shape[1])]
        per_joint = {
            name: float(delta)
            for name, delta in zip(trajectory.joint_names, deltas)
        }
        return float(np.max(np.abs(deltas))), per_joint

    def validate_trajectory_for_execution(self, trajectory):
        max_delta, per_joint = self.trajectory_joint_delta(trajectory)
        if max_delta > self.args.max_joint_delta:
            details = ", ".join(f"{name}={delta:+.3f}" for name, delta in per_joint.items())
            raise RuntimeError(
                "Refusing to execute: planned joint motion is too large "
                f"(max {max_delta:.3f} rad > limit {self.args.max_joint_delta:.3f} rad). "
                f"Joint deltas: {details}"
            )
        return max_delta, per_joint

    def execute_planned_trajectory(self, trajectory):
        if not trajectory.joint_names or not trajectory.points:
            raise RuntimeError("MoveGroup returned an empty trajectory")
        self.validate_trajectory_for_execution(trajectory)
        self.get_logger().info(f"Waiting for trajectory action server: {self.args.trajectory_action}")
        if not self.trajectory_client.wait_for_server(timeout_sec=self.args.action_wait_timeout):
            raise RuntimeError(f"Trajectory action server unavailable: {self.args.trajectory_action}")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Joint trajectory controller rejected the trajectory")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            raise RuntimeError("Joint trajectory controller returned no result")
        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                "Joint trajectory execution failed, "
                f"error_code={result.result.error_code}, error_string={result.result.error_string}"
            )
        return result.result

    def _position_constraints(self, position):
        constraints = Constraints()
        constraints.name = "red_target_handeye_approach"

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.args.goal_tolerance]

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.w = 1.0

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.args.base_frame
        position_constraint.link_name = self.args.end_effector_link
        position_constraint.constraint_region.primitives.append(sphere)
        position_constraint.constraint_region.primitive_poses.append(pose)
        position_constraint.weight = 1.0
        constraints.position_constraints.append(position_constraint)
        local_joint_limit = getattr(self.args, "local_joint_limit", self.args.max_joint_delta)
        if local_joint_limit > 0.0:
            with self.lock:
                joint_positions = dict(self.latest_joint_positions)
            for index in range(1, 7):
                name = f"joint{index}"
                if name not in joint_positions:
                    raise RuntimeError(f"Cannot constrain local IK solution: missing {name} state")
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = name
                joint_constraint.position = joint_positions[name]
                joint_constraint.tolerance_above = local_joint_limit
                joint_constraint.tolerance_below = local_joint_limit
                joint_constraint.weight = 1.0
                constraints.joint_constraints.append(joint_constraint)
        if self.args.orientation_weight > 0.0:
            try:
                base_t_gripper = self.lookup_base_t_gripper(None)
                quat = Rotation.from_matrix(base_t_gripper[:3, :3]).as_quat()
            except RuntimeError as exc:
                raise RuntimeError(f"Cannot build orientation constraint from current gripper pose: {exc}") from exc

            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = self.args.base_frame
            orientation_constraint.link_name = self.args.end_effector_link
            orientation_constraint.orientation.x = float(quat[0])
            orientation_constraint.orientation.y = float(quat[1])
            orientation_constraint.orientation.z = float(quat[2])
            orientation_constraint.orientation.w = float(quat[3])
            orientation_constraint.absolute_x_axis_tolerance = self.args.orientation_tolerance
            orientation_constraint.absolute_y_axis_tolerance = self.args.orientation_tolerance
            orientation_constraint.absolute_z_axis_tolerance = self.args.orientation_tolerance
            orientation_constraint.weight = self.args.orientation_weight
            constraints.orientation_constraints.append(orientation_constraint)
        return constraints

    def _show_preview(self, image, detection, target):
        if self.args.no_preview or self.preview_failed:
            return
        overlay = image.copy()
        if detection.contour is not None:
            cv2.drawContours(overlay, [detection.contour], -1, (0, 255, 0), 2)
        elif detection.bbox is not None:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if target is not None:
            u, v, depth_m, camera_point = target
            cv2.circle(overlay, (int(round(u)), int(round(v))), 5, (255, 255, 255), -1)
            text = (
                f"{detection.label or 'target'} u={u:.1f} v={v:.1f} z={depth_m:.3f}m "
                f"cam=({camera_point[0]:.3f},{camera_point[1]:.3f},{camera_point[2]:.3f})"
            )
        else:
            text = f"no target ({detection.reason})"
        cv2.putText(overlay, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        try:
            cv2.imshow("red target handeye approach", overlay)
            if detection.mask is not None:
                cv2.imshow("red target mask", detection.mask)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.preview_failed = True
            self.get_logger().warning(
                f"OpenCV preview is unavailable: {exc}. "
                f"Use rqt_image_view for {self.args.image_topic}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a red target and move the Z1 gripper near it to test hand-eye calibration."
    )
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument(
        "--depth-topic",
        default="/camera/camera/depth/image_rect_raw",
        help="Depth image topic in meters or uint16 millimeters. Set '' to use --target-distance.",
    )
    parser.add_argument(
        "--target-distance",
        type=float,
        default=0.45,
        help="Fallback target depth in meters when no usable depth is available.",
    )
    parser.add_argument("--base-frame", default="link00")
    parser.add_argument("--gripper-frame", default="gripperStator")
    parser.add_argument("--end-effector-link", default="gripperStator")
    parser.add_argument("--group", default="z1_arm")
    parser.add_argument("--move-group-action", default="/move_action")
    parser.add_argument("--trajectory-action", default="/joint_trajectory_controller/follow_joint_trajectory")
    parser.add_argument(
        "--mode",
        default="image-servo",
        choices=["image-servo", "ray"],
        help="image-servo centers the red target in the wrist camera; ray uses estimated 3D target position.",
    )
    parser.add_argument("--approach-distance", type=float, default=0.10, help="Stop this far before the target along the camera ray.")
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="Move this far from the current gripper toward the target. Set 0 to command the absolute approach point.",
    )
    parser.add_argument("--z-offset", type=float, default=0.0, help="Additional base-frame Z offset for the approach point.")
    parser.add_argument("--servo-step-size", type=float, default=0.005, help="Maximum lateral image-servo step in meters.")
    parser.add_argument("--servo-gain", type=float, default=0.02, help="Meters per normalized image error before clamping.")
    parser.add_argument("--servo-deadband-px", type=float, default=35.0, help="Pixel deadband considered centered.")
    parser.add_argument("--servo-forward-step", type=float, default=0.0, help="Forward optical-axis step when target is centered.")
    parser.add_argument("--goal-tolerance", type=float, default=0.008)
    parser.add_argument("--orientation-tolerance", type=float, default=0.08)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--max-joint-delta", type=float, default=0.12)
    parser.add_argument("--speed-scale", type=float, default=0.05)
    parser.add_argument("--planning-time", type=float, default=8.0)
    parser.add_argument("--planning-attempts", type=int, default=5)
    parser.add_argument("--action-wait-timeout", type=float, default=15.0)
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    parser.add_argument("--detect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--wait-for-trigger",
        action="store_true",
        help="Wait for a Trigger service call before detecting or planning.",
    )
    parser.add_argument("--trigger-service", default="/red_target/start")
    parser.add_argument(
        "--no-latest-tf-fallback",
        dest="allow_latest_tf_fallback",
        action="store_false",
        help="Fail instead of falling back to latest TF when image-time TF is just outside the buffer.",
    )
    parser.set_defaults(allow_latest_tf_fallback=True)
    parser.add_argument("--detect-only", action="store_true", help="Only detect and print target coordinates; do not call MoveIt.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the planned joint trajectory directly through joint_trajectory_controller.",
    )
    parser.add_argument("--no-preview", action="store_true")
    add_detector_arguments(parser)
    parser.add_argument("--min-area-px", type=float, default=200.0)
    parser.add_argument("--min-saturation", type=int, default=80)
    parser.add_argument("--min-value", type=int, default=50)
    parser.add_argument("--hue-high1", type=int, default=10)
    parser.add_argument("--hue-low2", type=int, default=170)
    parser.add_argument("--depth-window-px", type=int, default=25)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=2.5)
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    if args.depth_topic == "":
        args.depth_topic = None
    args.speed_scale = max(min(args.speed_scale, 1.0), 0.01)
    if not Path(args.calibration).exists():
        raise FileNotFoundError(f"Calibration file not found: {args.calibration}")
    if args.depth_topic is None and args.target_distance <= 0.0:
        raise ValueError("Set --depth-topic or provide --target-distance > 0")
    return args


def main():
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = RedTargetHandeyeApproach(args)
    try:
        node.wait_for_start_trigger()
        node.wait_for_inputs()
        camera_point, pixel, area, depth_m, stamp = node.wait_for_target()
        target_base, approach_base, _, _, _, _ = node.compute_approach_point(camera_point, stamp)
        centered = False
        if args.mode == "image-servo":
            motion_target, camera_origin_base, gripper_base, motion_mode, centered = node.compute_image_servo_point(pixel, stamp)
        else:
            (
                target_base,
                approach_base,
                motion_target,
                camera_origin_base,
                gripper_base,
                motion_mode,
            ) = node.compute_approach_point(camera_point, stamp)

        print("")
        print("Red target detection")
        print(f"  pixel:              u={pixel[0]:.1f}, v={pixel[1]:.1f}, area={area:.1f} px")
        print(f"  depth:              {depth_m:.4f} m")
        print(f"  target in camera:   x={camera_point[0]:.4f}, y={camera_point[1]:.4f}, z={camera_point[2]:.4f} m")
        print(
            "  camera in base:     "
            f"x={camera_origin_base[0]:.4f}, y={camera_origin_base[1]:.4f}, z={camera_origin_base[2]:.4f} m"
        )
        print(
            "  gripper in base:    "
            f"x={gripper_base[0]:.4f}, y={gripper_base[1]:.4f}, z={gripper_base[2]:.4f} m"
        )
        print(
            "  target in base:     "
            f"x={target_base[0]:.4f}, y={target_base[1]:.4f}, z={target_base[2]:.4f} m"
        )
        print(
            "  approach in base:   "
            f"x={approach_base[0]:.4f}, y={approach_base[1]:.4f}, z={approach_base[2]:.4f} m"
        )
        print(
            "  commanded target:   "
            f"x={motion_target[0]:.4f}, y={motion_target[1]:.4f}, z={motion_target[2]:.4f} m"
        )
        print(f"  motion mode:        {motion_mode}")
        if args.mode == "image-servo":
            print(f"  image centered:     {centered}")
        if args.detect_only:
            print("  mode:               detect only")
            return

        print(f"  mode:               {'execute' if args.execute else 'plan only'}")

        result = node.plan_move_group_goal(motion_target)
        node.publish_display_trajectory(result)
        rclpy.spin_once(node, timeout_sec=0.2)
        print("")
        print("MoveIt result")
        print(f"  success:            true")
        print(f"  planning_time:      {float(result.planning_time):.3f} s")
        print(f"  end_effector_link:  {args.end_effector_link}")
        print(f"  trajectory_points:  {len(result.planned_trajectory.joint_trajectory.points)}")
        max_delta, per_joint_delta = node.trajectory_joint_delta(result.planned_trajectory.joint_trajectory)
        print(f"  max_joint_delta:    {max_delta:.4f} rad")
        print(
            "  joint_deltas:       "
            + ", ".join(f"{name}={delta:+.3f}" for name, delta in per_joint_delta.items())
        )
        if args.execute:
            execution_result = node.execute_planned_trajectory(result.planned_trajectory.joint_trajectory)
            print("")
            print("Trajectory execution")
            print("  success:            true")
            if execution_result.error_string:
                print(f"  controller_msg:     {execution_result.error_string}")
        else:
            print("  note:               plan-only mode; add --execute to move the robot")
    except KeyboardInterrupt:
        print("\nInterrupted; no further command will be sent.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
