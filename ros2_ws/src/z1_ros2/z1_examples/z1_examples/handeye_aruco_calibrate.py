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


def aruco_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        choices = sorted(key for key in dir(cv2.aruco) if key.startswith("DICT_"))
        raise ValueError(f"Unknown ArUco dictionary '{name}'. Choices include: {', '.join(choices[:12])}")
    return cv2.aruco.Dictionary_get(getattr(cv2.aruco, name))


class HandEyeArucoNode(Node):
    def __init__(self, args):
        super().__init__("handeye_aruco_calibrate")
        self.args = args
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_image = None
        self.latest_stamp = None
        self.lock = threading.Lock()
        self.dictionary = aruco_dictionary(args.aruco_dictionary)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, args.camera_info_topic, self._camera_info_cb, 10)
        self.create_subscription(Image, args.image_topic, self._image_cb, 10)

    def _camera_info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _image_cb(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.latest_image = image
            self.latest_stamp = msg.header.stamp

    def detect_marker_pose(self):
        if self.camera_matrix is None:
            raise RuntimeError("No CameraInfo received yet")
        with self.lock:
            if self.latest_image is None:
                raise RuntimeError("No image received yet")
            image = self.latest_image.copy()

        corners, ids, _ = cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.parameters)
        if ids is None or len(ids) == 0:
            raise RuntimeError("No ArUco marker detected")

        ids_flat = ids.flatten()
        if self.args.marker_id >= 0:
            matches = np.where(ids_flat == self.args.marker_id)[0]
            if len(matches) == 0:
                raise RuntimeError(f"Marker id {self.args.marker_id} not detected; visible ids: {ids_flat.tolist()}")
            index = int(matches[0])
        else:
            index = 0

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[index]],
            self.args.marker_size,
            self.camera_matrix,
            self.dist_coeffs,
        )
        rotation, _ = cv2.Rodrigues(rvecs[0][0])
        translation = tvecs[0][0].reshape(3, 1)
        return rotation, translation, int(ids_flat[index])

    def lookup_gripper_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.base_frame,
                self.args.gripper_frame,
                Time(),
            )
        except TransformException as exc:
            raise RuntimeError(
                f"Cannot lookup TF {self.args.base_frame} <- {self.args.gripper_frame}: {exc}"
            ) from exc
        matrix = transform_to_matrix(transform)
        return matrix[:3, :3], matrix[:3, 3].reshape(3, 1)


def print_matrix(label: str, matrix: np.ndarray):
    print(label)
    for row in matrix:
        print("  " + " ".join(f"{value: .9f}" for value in row))


def write_yaml(path: Path, args, matrix: np.ndarray):
    xyz, quat = matrix_to_xyz_quat(matrix)
    data = {
        "parent_frame": args.gripper_frame,
        "child_frame": args.camera_frame,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive eye-in-hand calibration with a single ArUco marker."
    )
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--base-frame", default="world")
    parser.add_argument("--gripper-frame", default="gripperStator")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--marker-size", type=float, required=True, help="ArUco marker side length in meters.")
    parser.add_argument("--marker-id", type=int, default=-1, help="Marker id to use. Default: first visible marker.")
    parser.add_argument("--aruco-dictionary", default="DICT_4X4_50")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument(
        "--method",
        default="TSAI",
        choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
    )
    parser.add_argument("--output", default="/tmp/z1_handeye_gripper_to_camera.yaml")
    return parser.parse_args(remove_ros_args(args=sys.argv)[1:])


def main():
    args = parse_args()
    method = getattr(cv2, f"CALIB_HAND_EYE_{args.method}")

    rclpy.init(args=sys.argv)
    node = HandEyeArucoNode(args)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    r_gripper2base = []
    t_gripper2base = []
    r_target2cam = []
    t_target2cam = []

    print("")
    print("Eye-in-hand ArUco calibration")
    print(f"  base frame:    {args.base_frame}")
    print(f"  gripper frame: {args.gripper_frame}")
    print(f"  camera frame:  {args.camera_frame}")
    print(f"  image topic:   {args.image_topic}")
    print(f"  marker size:   {args.marker_size:.6f} m")
    print("")
    print("Move the robot to a new pose where the marker is visible, then press Enter.")
    print("Use varied wrist orientations. Type 'q' and Enter to stop.")

    try:
        while len(r_gripper2base) < args.samples:
            text = input(f"[{len(r_gripper2base)}/{args.samples}] sample> ").strip().lower()
            if text in ("q", "quit", "exit"):
                break
            try:
                rg, tg = node.lookup_gripper_pose()
                rt, tt, marker_id = node.detect_marker_pose()
            except RuntimeError as exc:
                print(f"  skipped: {exc}")
                continue

            r_gripper2base.append(rg)
            t_gripper2base.append(tg)
            r_target2cam.append(rt)
            t_target2cam.append(tt)
            print(f"  accepted sample {len(r_gripper2base)} using marker id {marker_id}")

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
        write_yaml(Path(args.output), args, gripper_t_camera)
        print(f"\nWrote result to {args.output}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
