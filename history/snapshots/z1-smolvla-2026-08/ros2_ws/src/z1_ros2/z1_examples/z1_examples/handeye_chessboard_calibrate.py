#!/usr/bin/env python3

import argparse
import math
import sys
import threading
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


def matrix_to_xyz_quat(matrix: np.ndarray):
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    xyz = matrix[:3, 3]
    return xyz, quat


def rt_to_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation.reshape(3)
    return matrix


def stamp_to_float(stamp):
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_to_name(stamp):
    if stamp is None:
        return "nostamp"
    return f"{stamp.sec}_{stamp.nanosec:09d}"


def matrix_to_record(matrix: np.ndarray):
    xyz, quat = matrix_to_xyz_quat(matrix)
    return {
        "translation": {
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        },
        "rotation_xyzw": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
        "matrix": matrix.tolist(),
    }


class HandEyeChessboardNode(Node):
    def __init__(self, args):
        super().__init__("handeye_chessboard_calibrate")
        self.args = args
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None
        self.latest_image_stamp = None
        self.lock = threading.Lock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.preview_enabled = args.preview
        self.preview_failed = False

        self.pattern_size = (args.inner_cols, args.inner_rows)
        self.object_points = self._make_object_points()

        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(Image, args.image_topic, self._image_cb, 10)
        if self.preview_enabled:
            self.create_timer(0.05, self._preview_cb)

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

    def _preview_cb(self):
        if self.preview_failed:
            return
        with self.lock:
            if self.latest_image is None:
                return
            image = self.latest_image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
        if found:
            cv2.drawChessboardCorners(image, self.pattern_size, corners, found)
            status = f"chessboard found: {self.args.inner_cols}x{self.args.inner_rows}"
            color = (0, 220, 0)
        else:
            status = f"looking for inner corners: {self.args.inner_cols}x{self.args.inner_rows}"
            color = (0, 0, 255)

        cv2.putText(
            image,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        try:
            cv2.imshow("z1 handeye chessboard preview", image)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.preview_failed = True
            self.get_logger().warning(
                f"OpenCV preview is unavailable: {exc}. "
                f"Use: ros2 run rqt_image_view rqt_image_view {self.args.image_topic}"
            )

    def detect_board_pose(self):
        if self.camera_matrix is None:
            raise RuntimeError("No CameraInfo received yet")
        with self.lock:
            if self.latest_image is None:
                raise RuntimeError("No image received yet")
            image = self.latest_image.copy()
            image_stamp = self.latest_image_stamp

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
        if not found:
            raise RuntimeError(
                f"Chessboard not detected. Expected inner corners {self.args.inner_cols}x{self.args.inner_rows}"
            )

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
            raise RuntimeError("solvePnP failed for detected chessboard")

        rotation, _ = cv2.Rodrigues(rvec)
        reproj, _ = cv2.projectPoints(
            self.object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        error = np.linalg.norm(refined.reshape(-1, 2) - reproj.reshape(-1, 2), axis=1).mean()
        return rotation, tvec.reshape(3, 1), float(error), image_stamp, image, refined, rvec, tvec

    def lookup_gripper_pose(self, stamp=None):
        use_image_time = self.args.tf_time_mode == "image" and stamp is not None
        tf_time = Time.from_msg(stamp) if use_image_time else Time()
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.base_frame,
                self.args.gripper_frame,
                tf_time,
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            if use_image_time:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.args.base_frame,
                        self.args.gripper_frame,
                        Time(),
                        timeout=Duration(seconds=0.2),
                    )
                except TransformException:
                    pass
                else:
                    self.get_logger().warning(
                        "TF at image time is unavailable; using latest TF instead. "
                        "Only sample while the robot is stationary."
                    )
                    matrix = transform_to_matrix(transform)
                    return matrix[:3, :3], matrix[:3, 3].reshape(3, 1)
            raise RuntimeError(
                f"Cannot lookup TF {self.args.base_frame} <- {self.args.gripper_frame} at image time: {exc}"
            ) from exc
        matrix = transform_to_matrix(transform)
        return matrix[:3, :3], matrix[:3, 3].reshape(3, 1)


def print_matrix(label: str, matrix: np.ndarray):
    print(label)
    for row in matrix:
        print("  " + " ".join(f"{value: .9f}" for value in row))


def write_yaml(path: Path, args, matrix: np.ndarray, sample_count: int):
    xyz, quat = matrix_to_xyz_quat(matrix)
    data = {
        "parent_frame": args.gripper_frame,
        "child_frame": args.camera_frame,
        "sample_count": sample_count,
        "chessboard_inner_corners": {"cols": args.inner_cols, "rows": args.inner_rows},
        "square_size_m": args.square_size,
        "translation": {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])},
        "rotation_xyzw": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
        "matrix": matrix.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def save_sample_image(
    output_dir: Path,
    index: int,
    image,
    corners,
    rvec,
    tvec,
    node: HandEyeChessboardNode,
    reprojection_error: float,
    image_stamp,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    cv2.drawChessboardCorners(annotated, node.pattern_size, corners, True)

    corner_points = corners.reshape(-1, 2)
    labels = {
        0: "0",
        node.args.inner_cols - 1: "x",
        (node.args.inner_rows - 1) * node.args.inner_cols: "y",
        len(corner_points) - 1: "last",
    }
    for corner_index, label in labels.items():
        if 0 <= corner_index < len(corner_points):
            x, y = corner_points[corner_index]
            cv2.circle(annotated, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                label,
                (int(round(x)) + 8, int(round(y)) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    axis_length = max(float(node.args.square_size) * 3.0, 0.03)
    try:
        cv2.drawFrameAxes(
            annotated,
            node.camera_matrix,
            node.dist_coeffs,
            rvec,
            tvec,
            axis_length,
            3,
        )
    except cv2.error:
        pass

    stamp_text = f"stamp {stamp_to_float(image_stamp):.6f}" if image_stamp is not None else "stamp none"
    cv2.putText(
        annotated,
        f"sample {index:02d}  reproj {reprojection_error:.3f}px  {stamp_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )

    filename = output_dir / f"sample_{index:02d}_{stamp_to_name(image_stamp)}.png"
    if not cv2.imwrite(str(filename), annotated):
        raise RuntimeError(f"Failed to write sample image: {filename}")
    return filename


def compute_sample_residuals(samples, gripper_t_camera: np.ndarray):
    base_t_targets = []
    for sample in samples:
        base_t_targets.append(
            sample["base_t_gripper"] @ gripper_t_camera @ sample["camera_t_board"]
        )

    positions = np.array([matrix[:3, 3] for matrix in base_t_targets])
    mean_position = positions.mean(axis=0)
    translation_errors = np.linalg.norm(positions - mean_position, axis=1)

    rotations = Rotation.from_matrix([matrix[:3, :3] for matrix in base_t_targets])
    mean_rotation = rotations.mean()
    rotation_errors_deg = []
    for rotation in rotations:
        delta = mean_rotation.inv() * rotation
        rotation_errors_deg.append(math.degrees(delta.magnitude()))

    return base_t_targets, mean_position, mean_rotation, translation_errors, rotation_errors_deg


def write_record_yaml(path: Path, args, gripper_t_camera: np.ndarray, samples):
    (
        base_t_boards,
        mean_position,
        mean_rotation,
        translation_errors,
        rotation_errors_deg,
    ) = compute_sample_residuals(samples, gripper_t_camera)

    rms_translation = math.sqrt(float(np.mean(translation_errors * translation_errors)))
    max_translation = float(np.max(translation_errors))
    rms_rotation = math.sqrt(float(np.mean(np.square(rotation_errors_deg))))
    max_rotation = float(np.max(rotation_errors_deg))

    mean_base_t_board = np.eye(4)
    mean_base_t_board[:3, :3] = mean_rotation.as_matrix()
    mean_base_t_board[:3, 3] = mean_position

    sample_records = []
    for index, sample in enumerate(samples):
        sample_records.append({
            "index": index + 1,
            "image_stamp_sec": stamp_to_float(sample["image_stamp"]),
            "reprojection_error_px": float(sample["reprojection_error_px"]),
            "base_frame": args.base_frame,
            "gripper_frame": args.gripper_frame,
            "camera_frame": args.camera_frame,
            "board_frame": "chessboard",
            "base_T_gripper": matrix_to_record(sample["base_t_gripper"]),
            "camera_T_board": matrix_to_record(sample["camera_t_board"]),
            "base_T_board_after_calibration": matrix_to_record(base_t_boards[index]),
            "residual_to_mean": {
                "translation_m": float(translation_errors[index]),
                "translation_mm": float(translation_errors[index] * 1000.0),
                "rotation_deg": float(rotation_errors_deg[index]),
            },
        })

    data = {
        "description": "Per-sample hand-eye chessboard calibration diagnostics.",
        "base_frame": args.base_frame,
        "gripper_frame": args.gripper_frame,
        "camera_frame": args.camera_frame,
        "board_frame": "chessboard",
        "method": args.method,
        "sample_count": len(samples),
        "chessboard_squares": {"cols": args.board_cols, "rows": args.board_rows},
        "chessboard_inner_corners": {"cols": args.inner_cols, "rows": args.inner_rows},
        "square_size_m": args.square_size,
        "gripper_T_camera": matrix_to_record(gripper_t_camera),
        "mean_base_T_board": matrix_to_record(mean_base_t_board),
        "residual_summary": {
            "translation_rms_m": rms_translation,
            "translation_rms_mm": rms_translation * 1000.0,
            "translation_max_m": max_translation,
            "translation_max_mm": max_translation * 1000.0,
            "rotation_rms_deg": rms_rotation,
            "rotation_max_deg": max_rotation,
        },
        "samples": sample_records,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data["residual_summary"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive eye-in-hand calibration with a chessboard target."
    )
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--base-frame", default="world")
    parser.add_argument("--gripper-frame", default="gripperStator")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument(
        "--tf-time-mode",
        default="latest",
        choices=["latest", "image"],
        help=(
            "TF lookup time for base<-gripper. Use 'latest' for manual stationary "
            "sampling; use 'image' only when camera and robot clocks are synchronized."
        ),
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        default=7,
        help="Number of chessboard squares along columns. Default: 7.",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=9,
        help="Number of chessboard squares along rows. Default: 9.",
    )
    parser.add_argument(
        "--inner-cols",
        type=int,
        default=None,
        help="Number of inner corners along columns. Overrides --board-cols - 1.",
    )
    parser.add_argument(
        "--inner-rows",
        type=int,
        default=None,
        help="Number of inner corners along rows. Overrides --board-rows - 1.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.025,
        help="Chessboard square side length in meters. Default: 0.025.",
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Disable OpenCV camera preview window.",
    )
    parser.set_defaults(preview=True)
    parser.add_argument(
        "--method",
        default="TSAI",
        choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
    )
    parser.add_argument("--output", default="/tmp/z1_handeye_chessboard_gripper_to_camera.yaml")
    parser.add_argument(
        "--record-output",
        default="/tmp/z1_handeye_chessboard_samples.yaml",
        help="Write per-sample poses and residual diagnostics to this YAML file.",
    )
    parser.add_argument(
        "--image-output-dir",
        default="/tmp/z1_handeye_chessboard_images",
        help="Directory for annotated accepted-sample images.",
    )
    parser.add_argument(
        "--no-save-images",
        dest="save_images",
        action="store_false",
        help="Do not save annotated accepted-sample images.",
    )
    parser.set_defaults(save_images=True)
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    args.inner_cols = args.inner_cols if args.inner_cols is not None else args.board_cols - 1
    args.inner_rows = args.inner_rows if args.inner_rows is not None else args.board_rows - 1
    if args.inner_cols <= 1 or args.inner_rows <= 1:
        raise ValueError("Inner corner count must be greater than 1 in both directions")
    return args


def main():
    args = parse_args()
    method = getattr(cv2, f"CALIB_HAND_EYE_{args.method}")

    rclpy.init(args=sys.argv)
    node = HandEyeChessboardNode(args)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    r_gripper2base = []
    t_gripper2base = []
    r_target2cam = []
    t_target2cam = []
    samples = []

    print("")
    print("Eye-in-hand chessboard calibration")
    print(f"  base frame:        {args.base_frame}")
    print(f"  gripper frame:     {args.gripper_frame}")
    print(f"  camera frame:      {args.camera_frame}")
    print(f"  image topic:       {args.image_topic}")
    print(f"  chessboard squares:{args.board_cols}x{args.board_rows}")
    print(f"  inner corners:     {args.inner_cols}x{args.inner_rows}")
    print(f"  square size:       {args.square_size:.6f} m")
    print(f"  preview:           {'enabled' if args.preview else 'disabled'}")
    print(f"  save images:       {'enabled' if args.save_images else 'disabled'}")
    if args.save_images:
        print(f"  image output dir:  {args.image_output_dir}")
    print("")
    print("Move the robot to a pose where the whole chessboard is visible, then press Enter.")
    print("Use varied wrist orientations and keep the target fixed. Type 'q' and Enter to stop.")

    try:
        while len(r_gripper2base) < args.samples:
            text = input(f"[{len(r_gripper2base)}/{args.samples}] sample> ").strip().lower()
            if text in ("q", "quit", "exit"):
                break
            try:
                rt, tt, error, image_stamp, image, corners, rvec, tvec = node.detect_board_pose()
                rg, tg = node.lookup_gripper_pose(image_stamp)
            except RuntimeError as exc:
                print(f"  skipped: {exc}")
                continue

            sample_index = len(r_gripper2base) + 1
            r_gripper2base.append(rg)
            t_gripper2base.append(tg)
            r_target2cam.append(rt)
            t_target2cam.append(tt)
            samples.append({
                "image_stamp": image_stamp,
                "reprojection_error_px": error,
                "base_t_gripper": rt_to_matrix(rg, tg),
                "camera_t_board": rt_to_matrix(rt, tt),
            })
            image_path = None
            if args.save_images:
                try:
                    image_path = save_sample_image(
                        Path(args.image_output_dir),
                        sample_index,
                        image,
                        corners,
                        rvec,
                        tvec,
                        node,
                        error,
                        image_stamp,
                    )
                except RuntimeError as exc:
                    print(f"  warning: {exc}")

            message = f"  accepted sample {len(r_gripper2base)}; reprojection error {error:.3f} px"
            if image_path is not None:
                message += f"; image {image_path}"
            print(message)

        if len(r_gripper2base) < 3:
            raise RuntimeError("Need at least 3 accepted samples to calibrate")

        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=method,
        )

        gripper_t_camera = np.eye(4)
        gripper_t_camera[:3, :3] = r_cam2gripper
        gripper_t_camera[:3, 3] = t_cam2gripper.reshape(3)
        xyz, quat = matrix_to_xyz_quat(gripper_t_camera)

        print("")
        print_matrix(f"{args.gripper_frame} -> {args.camera_frame} matrix:", gripper_t_camera)
        print("")
        print("static_transform_publisher command:")
        print(
            "ros2 run tf2_ros static_transform_publisher "
            f"{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
            f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f} "
            f"{args.gripper_frame} {args.camera_frame}"
        )
        residual_summary = write_record_yaml(Path(args.record_output), args, gripper_t_camera, samples)
        print("")
        print(
            "sample residuals: "
            f"translation RMS {residual_summary['translation_rms_mm']:.2f} mm, "
            f"max {residual_summary['translation_max_mm']:.2f} mm; "
            f"rotation RMS {residual_summary['rotation_rms_deg']:.3f} deg, "
            f"max {residual_summary['rotation_max_deg']:.3f} deg"
        )
        write_yaml(Path(args.output), args, gripper_t_camera, len(r_gripper2base))
        print(f"\nWrote result to {args.output}")
        print(f"Wrote per-sample diagnostics to {args.record_output}")
        if args.save_images:
            print(f"Wrote accepted-sample images to {args.image_output_dir}")
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
