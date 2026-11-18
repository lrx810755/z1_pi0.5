#!/usr/bin/env python3

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformException, TransformListener


def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def rt_to_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation.reshape(3)
    return matrix


def load_matrix(path: Path) -> np.ndarray:
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


class HandEyeChessboardValidator(Node):
    def __init__(self, args, gripper_t_camera: np.ndarray):
        super().__init__("handeye_chessboard_validate")
        self.args = args
        self.gripper_t_camera = gripper_t_camera
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None
        self.latest_image_stamp = None
        self.lock = threading.Lock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pattern_size = (args.inner_cols, args.inner_rows)
        self.object_points = self._make_object_points()
        self.samples = []
        self.preview_failed = False

        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(Image, args.image_topic, self._image_cb, 10)
        self.create_timer(args.period, self._validate_cb)

    def _make_object_points(self):
        points = np.zeros((self.args.inner_cols * self.args.inner_rows, 3), np.float32)
        grid = np.mgrid[0:self.args.inner_cols, 0:self.args.inner_rows].T.reshape(-1, 2)
        points[:, :2] = grid * self.args.square_size
        return points

    def _camera_info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _image_cb(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = msg.header.stamp

    def _detect_board_pose(self, image):
        if self.camera_matrix is None:
            raise RuntimeError("No CameraInfo received yet")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
        if not found:
            raise RuntimeError("Chessboard not detected")
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.001,
        )
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            refined,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed")
        rotation, _ = cv2.Rodrigues(rvec)
        reproj, _ = cv2.projectPoints(
            self.object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        error = np.linalg.norm(refined.reshape(-1, 2) - reproj.reshape(-1, 2), axis=1).mean()
        return rotation, tvec.reshape(3, 1), refined, error

    def _lookup_base_t_gripper(self, stamp=None):
        tf_time = Time.from_msg(stamp) if stamp is not None else Time()
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.base_frame,
                self.args.gripper_frame,
                tf_time,
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            raise RuntimeError(
                f"Cannot lookup TF {self.args.base_frame} <- {self.args.gripper_frame} at image time: {exc}"
            ) from exc
        return transform_to_matrix(transform)

    def _validate_cb(self):
        with self.lock:
            if self.latest_image is None:
                return
            image = self.latest_image.copy()
            image_stamp = self.latest_image_stamp

        try:
            r_target2cam, t_target2cam, corners, reproj_error = self._detect_board_pose(image)
            base_t_gripper = self._lookup_base_t_gripper(image_stamp)
        except RuntimeError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=2.0)
            self._show_preview(image, None, None)
            return

        camera_t_target = rt_to_matrix(r_target2cam, t_target2cam)
        base_t_target = base_t_gripper @ self.gripper_t_camera @ camera_t_target
        target_position = base_t_target[:3, 3]
        self.samples.append(target_position)
        if len(self.samples) > self.args.window:
            self.samples.pop(0)

        positions = np.array(self.samples)
        mean = positions.mean(axis=0)
        distances = np.linalg.norm(positions - mean, axis=1)
        rms = math.sqrt(float(np.mean(distances * distances))) if len(distances) else 0.0
        max_error = float(distances.max()) if len(distances) else 0.0

        self.get_logger().info(
            "board in %s: xyz=(%.4f, %.4f, %.4f) m, window_rms=%.2f mm, max=%.2f mm, reproj=%.3f px"
            % (
                self.args.base_frame,
                target_position[0],
                target_position[1],
                target_position[2],
                rms * 1000.0,
                max_error * 1000.0,
                reproj_error,
            )
        )
        self._show_preview(image, corners, (rms, max_error, reproj_error))

    def _show_preview(self, image, corners, metrics):
        if self.args.no_preview or self.preview_failed:
            return
        if corners is not None:
            cv2.drawChessboardCorners(image, self.pattern_size, corners, True)
        if metrics is None:
            text = "chessboard not detected"
            color = (0, 0, 255)
        else:
            rms, max_error, reproj_error = metrics
            text = f"RMS {rms * 1000.0:.1f} mm  MAX {max_error * 1000.0:.1f} mm  reproj {reproj_error:.2f}px"
            color = (0, 220, 0)
        cv2.putText(image, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        try:
            cv2.imshow("z1 handeye validation", image)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.preview_failed = True
            self.get_logger().warning(
                f"OpenCV preview is unavailable: {exc}. "
                f"Use: ros2 run rqt_image_view rqt_image_view {self.args.image_topic}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate an eye-in-hand chessboard calibration by checking target stability in the base frame."
    )
    parser.add_argument("--calibration", default="/tmp/z1_handeye_chessboard_gripper_to_camera.yaml")
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--base-frame", default="world")
    parser.add_argument("--gripper-frame", default="gripperStator")
    parser.add_argument("--board-cols", type=int, default=7)
    parser.add_argument("--board-rows", type=int, default=9)
    parser.add_argument("--inner-cols", type=int, default=None)
    parser.add_argument("--inner-rows", type=int, default=None)
    parser.add_argument("--square-size", type=float, default=0.025)
    parser.add_argument("--period", type=float, default=1.0, help="Validation update period in seconds.")
    parser.add_argument("--window", type=int, default=30, help="Number of recent detections used for RMS stability.")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    args.inner_cols = args.inner_cols if args.inner_cols is not None else args.board_cols - 1
    args.inner_rows = args.inner_rows if args.inner_rows is not None else args.board_rows - 1
    return args


def main():
    args = parse_args()
    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    gripper_t_camera = load_matrix(calibration_path)

    print("")
    print("Eye-in-hand validation")
    print(f"  calibration:   {calibration_path}")
    print(f"  base frame:    {args.base_frame}")
    print(f"  gripper frame: {args.gripper_frame}")
    print(f"  inner corners: {args.inner_cols}x{args.inner_rows}")
    print("")
    print("Keep the chessboard fixed. Move the robot and watch window_rms/max.")
    print("A good calibration keeps board xyz nearly constant in the base frame.")

    rclpy.init(args=sys.argv)
    node = HandEyeChessboardValidator(args, gripper_t_camera)
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
