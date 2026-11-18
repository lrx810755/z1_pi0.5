#!/usr/bin/env python3

import argparse
import copy
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, JointState
from scipy.spatial.transform import Rotation

from z1_examples.red_target_handeye_approach import RedTargetHandeyeApproach
from z1_examples.target_detector import DetectorConfig, TargetDetector, add_detector_arguments


DEFAULT_CALIBRATION = Path.home() / "ros2_ws" / "z1_handeye_charuco_gripper_to_camera.yaml"
DEFAULT_GRIPPER_OPEN_RAD = -0.90
# Lower the bottle approach by 50 mm from the previous +25 mm setting.
# Keep this explicit and overridable so future trials can recalibrate it.
DEFAULT_BOTTLE_Z_OFFSET_M = -0.025
DEFAULT_BOTTLE_Y_OFFSET_M = 0.02
DEFAULT_GRIPPER_CLOSE_DEG = 25.0
DEFAULT_INSERTION_OFFSET_M = 0.03
DEFAULT_GRASP_PITCH_DEG = 5.0
GRASP_PITCH_TOLERANCE_DEG = 1.0
HOME_JOINT_TARGETS = {f"joint{index}": 0.0 for index in range(1, 7)}
HOME_POSITION_TOLERANCE_RAD = 0.05
HOME_VELOCITY_TOLERANCE_RAD_S = 0.08
HOME_SETTLE_TIMEOUT_S = 2.0
HOME_SETTLE_SAMPLES = 3
AUTO_SEQUENCE_NEXT_ACTION = {
    "approach": "grasp",
    "grasp": "home",
    "home": "release",
}


def horizontal_grasp_orientation(approach_direction, pitch_degrees=DEFAULT_GRASP_PITCH_DEG):
    """Build an XYZW reference whose local +X inserts toward the bottle."""
    direction = np.asarray(approach_direction, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError("抓取接近方向必须是三维向量")
    horizontal = direction.copy()
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-6:
        raise ValueError("无法从竖直接近方向确定水平抓取朝向")
    horizontal /= norm

    pitch = np.deg2rad(float(pitch_degrees))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cos(pitch) * horizontal + np.sin(pitch) * world_up
    y_axis = np.cross(world_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return Rotation.from_matrix(
        np.column_stack((x_axis, y_axis, z_axis))
    ).as_quat()


def offset_along_tool_x(position, orientation_quat, offset_m):
    """Advance a Cartesian target along the quaternion's local +X axis."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("插入目标位置必须是三维向量")
    offset_m = float(offset_m)
    if offset_m < 0.0:
        raise ValueError("插入深入量不能为负数")
    tool_x = Rotation.from_quat(orientation_quat).apply([1.0, 0.0, 0.0])
    return position + tool_x * offset_m


def offset_along_base_y(position, offset_m):
    """Apply a final Cartesian offset along link00/base-frame Y."""
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("Y轴偏移目标位置必须是三维向量")
    result = position.copy()
    result[1] += float(offset_m)
    return result


def gripper_close_target_rad(close_degrees):
    """Convert a positive user-facing closing angle to the Z1 joint sign."""
    close_degrees = float(close_degrees)
    if not 0.0 <= close_degrees <= 90.0:
        raise ValueError("夹爪闭合角必须在 [0°, 90°] 内")
    return float(np.deg2rad(-close_degrees))


@dataclass
class Detection:
    found: bool
    u: float = 0.0
    v: float = 0.0
    area: float = 0.0
    depth: float = 0.0


class EmergencyStop(RuntimeError):
    """Raised after an active trajectory has been cancelled by the GUI E-stop."""


class TrajectoryExecutionError(RuntimeError):
    def __init__(self, error_code):
        self.error_code = int(error_code)
        super().__init__(f"MoveIt 轨迹执行失败，error_code={self.error_code}")


def _duration_seconds(duration):
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def add_gripper_motion(robot_trajectory, start_position, target_position, joint_name):
    """Add a smooth, time-synchronised gripper motion to a MoveIt trajectory."""
    trajectory = robot_trajectory.joint_trajectory
    if joint_name in trajectory.joint_names:
        raise RuntimeError(f"轨迹中已包含夹爪关节 {joint_name}")
    if not trajectory.points:
        raise RuntimeError("无法向空轨迹添加夹爪动作")
    start_time = _duration_seconds(trajectory.points[0].time_from_start)
    total_time = _duration_seconds(trajectory.points[-1].time_from_start) - start_time
    if total_time <= 0.0:
        raise RuntimeError("规划轨迹持续时间无效")

    delta = float(target_position) - float(start_position)
    trajectory.joint_names.append(joint_name)
    for point in trajectory.points:
        elapsed = _duration_seconds(point.time_from_start) - start_time
        phase = min(1.0, max(0.0, elapsed / total_time))
        blend = phase * phase * (3.0 - 2.0 * phase)
        point.positions.append(float(start_position) + delta * blend)
        if point.velocities:
            point.velocities.append(delta * 6.0 * phase * (1.0 - phase) / total_time)
        if point.accelerations:
            point.accelerations.append(
                delta * 6.0 * (1.0 - 2.0 * phase) / (total_time ** 2)
            )
        if point.effort:
            point.effort.append(0.0)
    return robot_trajectory


class CameraNode(Node):
    def __init__(self, args):
        super().__init__("red_target_gui")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.image = None
        self.depth = None
        self.camera_matrix = None
        self.image_stamp = None
        self.joint_positions = {}
        self.create_subscription(Image, args.image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(Image, args.depth_topic, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._info_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

    def _joint_state_cb(self, msg):
        with self.lock:
            self.joint_positions.update(
                {name: float(position) for name, position in zip(msg.name, msg.position)}
            )

    def joint_position(self, name):
        with self.lock:
            return self.joint_positions.get(name)

    def _image_cb(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.image = np.asarray(image).copy()
            self.image_stamp = msg.header.stamp

    def _depth_cb(self, msg):
        depth = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = depth.astype(np.float32)
        with self.lock:
            self.depth = depth.copy()

    def _info_cb(self, msg):
        with self.lock:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def snapshot(self):
        with self.lock:
            image = None if self.image is None else self.image.copy()
            depth = None if self.depth is None else self.depth.copy()
            matrix = None if self.camera_matrix is None else self.camera_matrix.copy()
            stamp = self.image_stamp
        return image, depth, matrix, stamp


class ApprovedTrajectoryExecutor(Node):
    def __init__(self, action_name="/execute_trajectory"):
        super().__init__("red_target_trajectory_executor")
        self.lock = threading.Lock()
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_state_generation = 0
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.client = ActionClient(self, ExecuteTrajectory, action_name)
        self.stop_requested = threading.Event()
        self.goal_handle = None

    def request_stop(self):
        self.stop_requested.set()

    def _spin_future_or_stop(self, future):
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.stop_requested.is_set():
                raise EmergencyStop("急停已触发，轨迹未启动")
        if self.stop_requested.is_set():
            raise EmergencyStop("急停已触发，轨迹未启动")
        return future.result()

    def _cancel_active_goal(self):
        if self.goal_handle is None:
            return
        cancel_future = self.goal_handle.cancel_goal_async()
        deadline = time.monotonic() + 3.0
        while rclpy.ok() and not cancel_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not cancel_future.done():
            raise EmergencyStop("急停已触发，但控制器取消确认超时；请按下实体急停")
        response = cancel_future.result()
        if response is None or not response.goals_canceling:
            raise EmergencyStop("急停取消请求未被控制器确认；请按下实体急停")
        raise EmergencyStop("急停已触发，控制器已接受轨迹取消请求")

    def _joint_state_cb(self, msg):
        with self.lock:
            self.joint_positions = {
                name: float(position) for name, position in zip(msg.name, msg.position)
            }
            self.joint_velocities = {
                name: float(velocity) for name, velocity in zip(msg.name, msg.velocity)
            }
            self.joint_state_generation += 1

    def wait_for_settled_goal(
        self,
        targets,
        position_tolerance,
        velocity_tolerance,
        timeout,
        required_samples,
    ):
        """Require several fresh joint states near a goal before accepting it."""
        names = tuple(targets)
        deadline = time.monotonic() + float(timeout)
        settled_samples = 0
        last_generation = -1
        worst_position = ("", float("inf"))
        worst_velocity = ("", float("inf"))
        while rclpy.ok() and time.monotonic() < deadline:
            if self.stop_requested.is_set():
                raise EmergencyStop("急停已触发，不能确认 HOME 状态")
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.lock:
                if self.joint_state_generation == last_generation:
                    continue
                last_generation = self.joint_state_generation
                if not all(
                    name in self.joint_positions and name in self.joint_velocities
                    for name in names
                ):
                    continue
                position_errors = {
                    name: abs(self.joint_positions[name] - float(targets[name]))
                    for name in names
                }
                speeds = {name: abs(self.joint_velocities[name]) for name in names}
            worst_position = max(position_errors.items(), key=lambda item: item[1])
            worst_velocity = max(speeds.items(), key=lambda item: item[1])
            if (
                worst_position[1] <= position_tolerance
                and worst_velocity[1] <= velocity_tolerance
            ):
                settled_samples += 1
                if settled_samples >= required_samples:
                    return worst_position, worst_velocity
            else:
                settled_samples = 0
        if worst_position[0]:
            raise RuntimeError(
                "HOME 未稳定到达："
                f"{worst_position[0]} 位置误差 {worst_position[1]:.4f} rad "
                f"(允许 {position_tolerance:.4f})，"
                f"{worst_velocity[0]} 速度 {worst_velocity[1]:.4f} rad/s "
                f"(允许 {velocity_tolerance:.4f})"
            )
        raise RuntimeError("HOME 超时后未收到完整的关节位置和速度状态")

    def validate_start(self, robot_trajectory, tolerance, timeout=5.0):
        trajectory = robot_trajectory.joint_trajectory
        if not trajectory.joint_names or not trajectory.points:
            raise RuntimeError("已审核轨迹为空")
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.stop_requested.is_set():
                raise EmergencyStop("急停已触发，轨迹未启动")
            rclpy.spin_once(self, timeout_sec=0.05)
            with self.lock:
                if all(name in self.joint_positions for name in trajectory.joint_names):
                    current = np.array(
                        [self.joint_positions[name] for name in trajectory.joint_names],
                        dtype=np.float64,
                    )
                    planned_start = np.array(trajectory.points[0].positions, dtype=np.float64)
                    errors = np.abs(current - planned_start)
                    max_error = float(np.max(errors))
                    if max_error > tolerance:
                        joint = trajectory.joint_names[int(np.argmax(errors))]
                        raise RuntimeError(
                            f"轨迹已失效：当前状态偏离规划起点 {max_error:.3f} rad "
                            f"({joint})，允许值 {tolerance:.3f} rad。请重新规划"
                        )
                    return max_error
        raise RuntimeError("未收到完整关节状态，无法验证轨迹起点")

    def execute(self, robot_trajectory, start_tolerance):
        start_error = self.validate_start(robot_trajectory, start_tolerance)
        if self.stop_requested.is_set():
            raise EmergencyStop("急停已触发，轨迹未启动")
        deadline = time.monotonic() + 10.0
        while not self.client.wait_for_server(timeout_sec=0.1):
            if self.stop_requested.is_set():
                raise EmergencyStop("急停已触发，轨迹未启动")
            if time.monotonic() >= deadline:
                raise RuntimeError("MoveIt /execute_trajectory action 不可用")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_trajectory
        future = self.client.send_goal_async(goal)
        handle = self._spin_future_or_stop(future)
        if handle is None or not handle.accepted:
            raise RuntimeError("MoveIt 拒绝执行已审核轨迹")
        self.goal_handle = handle
        result_future = handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.stop_requested.is_set():
                self._cancel_active_goal()
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError("MoveIt 执行没有返回结果")
        code = wrapped_result.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            raise TrajectoryExecutionError(code)
        return start_error


class WorkerSignals(QObject):
    plan_finished = pyqtSignal(bool, str, object)
    execution_finished = pyqtSignal(bool, str)
    detection_finished = pyqtSignal(object, str)


class RedTargetWindow(QMainWindow):
    def __init__(self, args, node):
        super().__init__()
        self.args = args
        self.node = node
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self.recognition_active = False
        self.last_detection = Detection(False)
        self.detector = TargetDetector(self._detector_config_from_args())
        self.detection_busy = False
        self.detection_generation = 0
        self.planning = False
        self.executing = False
        self.emergency_latched = False
        self.safety_generation = 0
        self.active_executor = None
        self.active_executor_lock = threading.Lock()
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.approved_action = None
        self._planning_action = None
        self.auto_sequence_active = False
        self.plan_feedback_until = 0.0
        self.started_at = time.monotonic()
        self.worker_signals = WorkerSignals()
        self.worker_signals.plan_finished.connect(self._planning_finished)
        self.worker_signals.execution_finished.connect(self._execution_finished)
        self.worker_signals.detection_finished.connect(self._detection_finished)
        self.setWindowTitle("Z1 红色目标识别")
        self.setMinimumSize(1040, 720)
        self.resize(1240, 900)
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title_row = QHBoxLayout()
        title = QLabel("Z1 红色目标识别")
        title.setObjectName("title")
        self.connection_label = QLabel("等待相机")
        self.connection_label.setObjectName("muted")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.connection_label)
        root.addLayout(title_row)

        content = QHBoxLayout()
        content.setSpacing(14)
        self.video = QLabel("等待彩色图像…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(600, 450)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setObjectName("video")
        content.addWidget(self.video, 1)

        side_scroll = QScrollArea()
        side_scroll.setObjectName("sideScroll")
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setFixedWidth(340)

        side = QFrame()
        side.setObjectName("side")
        side.setMinimumWidth(318)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)

        self.status = QLabel("识别已暂停")
        self.status.setObjectName("statusIdle")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setMinimumHeight(44)
        side_layout.addWidget(self.status)

        self.start_button = QPushButton("开始识别")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.clicked.connect(self._toggle_recognition)
        self.start_button.setMinimumHeight(40)
        side_layout.addWidget(self.start_button)

        detector_form = QFormLayout()
        self.detector_mode = QComboBox()
        self.detector_mode.addItem("YOLO追踪", "yolo")
        self.detector_mode.addItem("像素追踪", "pixel")
        initial_mode = "yolo" if getattr(self.args, "yolo", False) else self.args.detector
        self.detector_mode.setCurrentIndex(self.detector_mode.findData(initial_mode))
        self.model_path = QLineEdit(self.args.yolo_model)
        self.target_class = QComboBox()
        self.target_class.setEditable(True)
        self.target_class.addItem(self.args.yolo_class)
        self.yolo_confidence = QDoubleSpinBox()
        self.yolo_confidence.setRange(0.01, 1.0)
        self.yolo_confidence.setSingleStep(0.05)
        self.yolo_confidence.setValue(self.args.yolo_conf)
        self.apply_detector_button = QPushButton("应用识别参数")
        self.apply_detector_button.clicked.connect(self._apply_detector_config)
        detector_form.addRow("识别方式", self.detector_mode)
        detector_form.addRow("模型", self.model_path)
        detector_form.addRow("目标类别", self.target_class)
        detector_form.addRow("置信度", self.yolo_confidence)
        side_layout.addLayout(detector_form)
        side_layout.addWidget(self.apply_detector_button)
        self.detector_mode.currentIndexChanged.connect(self._detector_mode_changed)
        self._update_detector_controls()

        self.plan_button = QPushButton("规划完整接近轨迹")
        self.plan_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        self.plan_button.clicked.connect(self._plan)
        self.plan_button.setEnabled(False)
        self.plan_button.setMinimumHeight(40)
        side_layout.addWidget(self.plan_button)

        self.execute_button = QPushButton("执行已审核轨迹")
        self.execute_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.execute_button.clicked.connect(self._confirm_execute)
        self.execute_button.setEnabled(False)
        self.execute_button.setMinimumHeight(40)
        self.execute_button.setObjectName("executeButton")
        side_layout.addWidget(self.execute_button)

        self.grasp_button = QPushButton(
            f"规划夹爪闭合至 {self.args.gripper_close_deg:.0f}°"
        )
        self.grasp_button.clicked.connect(self._plan_grasp)
        self.grasp_button.setMinimumHeight(40)
        side_layout.addWidget(self.grasp_button)

        self.home_button = QPushButton("规划返回 HOME 位")
        self.home_button.clicked.connect(self._plan_home)
        self.home_button.setMinimumHeight(40)
        side_layout.addWidget(self.home_button)

        self.auto_sequence_button = QPushButton("一键抓取并回 HOME 放置")
        self.auto_sequence_button.clicked.connect(self._start_auto_sequence)
        self.auto_sequence_button.setEnabled(False)
        self.auto_sequence_button.setMinimumHeight(44)
        self.auto_sequence_button.setObjectName("autoSequenceButton")
        side_layout.addWidget(self.auto_sequence_button)

        self.emergency_button = QPushButton("急停（立即停止）")
        self.emergency_button.clicked.connect(self._emergency_stop)
        self.emergency_button.setMinimumHeight(48)
        self.emergency_button.setObjectName("emergencyButton")
        side_layout.addWidget(self.emergency_button)

        self.reset_emergency_button = QPushButton("复位急停")
        self.reset_emergency_button.clicked.connect(self._reset_emergency_stop)
        self.reset_emergency_button.setEnabled(False)
        side_layout.addWidget(self.reset_emergency_button)

        details = QFormLayout()
        details.setVerticalSpacing(10)
        self.center_value = QLabel("-")
        self.area_value = QLabel("-")
        self.depth_value = QLabel("-")
        self.detector_value = QLabel("像素追踪" if initial_mode == "pixel" else "YOLO追踪")
        details.addRow("目标中心", self.center_value)
        details.addRow("轮廓面积", self.area_value)
        details.addRow("目标深度", self.depth_value)
        details.addRow("实际识别", self.detector_value)
        side_layout.addLayout(details)
        side_layout.addStretch()

        note = QLabel(
            "手动模式先在 RViz 审核轨迹；一键模式确认一次后自动完成接近、抓取、"
            "回 HOME 和释放，任一步失败或急停都会中断。"
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        side_layout.addWidget(note)
        side_scroll.setWidget(side)
        content.addWidget(side_scroll)
        root.addLayout(content, 1)
        self.setCentralWidget(central)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6f8; color: #18212b; font-size: 14px; }
            QLabel#title { font-size: 22px; font-weight: 600; }
            QLabel#muted { color: #65717d; }
            QLabel#video { background: #101419; color: #aab3bc; border: 1px solid #c8d0d8; }
            QScrollArea#sideScroll { background: transparent; border: none; }
            QFrame#side { background: #ffffff; border: 1px solid #d6dde3; border-radius: 6px; }
            QLabel#statusIdle { background: #e8edf1; color: #4e5b66; border-radius: 4px; font-weight: 600; }
            QLabel#statusFound { background: #dff3e7; color: #176b3a; border-radius: 4px; font-weight: 600; }
            QLabel#statusMissing { background: #f9e3e3; color: #9c2929; border-radius: 4px; font-weight: 600; }
            QLabel#statusBusy { background: #fff1cf; color: #73520c; border-radius: 4px; font-weight: 600; }
            QPushButton { background: #ffffff; border: 1px solid #aeb9c3; border-radius: 4px; padding: 7px 10px; }
            QPushButton:hover { background: #eef3f7; }
            QPushButton:disabled { color: #9ba5ad; background: #edf0f2; }
            QPushButton#executeButton:enabled { background: #a52a2a; color: #ffffff; border-color: #842020; }
            QPushButton#executeButton:enabled:hover { background: #8f2222; }
            QPushButton#autoSequenceButton:enabled { background: #176b3a; color: #ffffff; border-color: #10562e; font-weight: 600; }
            QPushButton#autoSequenceButton:enabled:hover { background: #105c31; }
            QPushButton#emergencyButton { background: #d00000; color: #ffffff; border: 2px solid #830000; font-size: 16px; font-weight: 700; }
            QPushButton#emergencyButton:hover { background: #a80000; }
        """)

    def _emergency_stop(self):
        self.emergency_latched = True
        self.safety_generation += 1
        self.recognition_active = False
        self.start_button.setText("开始识别")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.setEnabled(False)
        self.plan_button.setEnabled(False)
        self.execute_button.setEnabled(False)
        self.grasp_button.setEnabled(False)
        self.home_button.setEnabled(False)
        self.auto_sequence_button.setEnabled(False)
        self.auto_sequence_active = False
        self.reset_emergency_button.setEnabled(not self.executing)
        self._invalidate_approved_trajectory_silent("急停已锁止；复位后才能重新规划")
        with self.active_executor_lock:
            executor_node = self.active_executor
        if executor_node is not None:
            executor_node.request_stop()
            self._set_status("急停：正在取消轨迹…", "statusMissing")
        else:
            self._set_status("急停已锁止", "statusMissing")
        self.statusBar().showMessage(
            "软件急停已触发；若机械臂仍在运动，请立即按下实体急停", 20000
        )

    def _reset_emergency_stop(self):
        if self.executing:
            return
        self.emergency_latched = False
        self.start_button.setEnabled(True)
        self.grasp_button.setEnabled(True)
        self.home_button.setEnabled(True)
        self._update_auto_sequence_button()
        self.reset_emergency_button.setEnabled(False)
        self._set_status("急停已复位，识别已暂停", "statusIdle")
        self.statusBar().showMessage("急停已复位，请重新识别并规划", 10000)

    def _toggle_recognition(self):
        if self.emergency_latched:
            return
        self.recognition_active = not self.recognition_active
        if self.recognition_active:
            self.start_button.setText("停止识别")
            self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        else:
            self.start_button.setText("开始识别")
            self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.last_detection = Detection(False)
            self.plan_feedback_until = 0.0
            self.plan_button.setEnabled(False)
            self._set_status("识别已暂停", "statusIdle")
        self._update_auto_sequence_button()

    def _detector_config_from_args(self):
        mode = "yolo" if getattr(self.args, "yolo", False) else self.args.detector
        return DetectorConfig(
            mode=mode,
            model_path=self.args.yolo_model,
            target_class=self.args.yolo_class,
            confidence=self.args.yolo_conf,
            min_area_px=self.args.min_area_px,
            min_saturation=self.args.min_saturation,
            min_value=self.args.min_value,
            hue_high1=self.args.hue_high1,
            hue_low2=self.args.hue_low2,
            device=self.args.yolo_device or None,
        )

    def _current_detector_config(self):
        return DetectorConfig(
            mode=self.detector_mode.currentData(),
            model_path=self.model_path.text().strip() or "yolo11n.pt",
            target_class=self.target_class.currentText().strip(),
            confidence=self.yolo_confidence.value(),
            min_area_px=self.args.min_area_px,
            min_saturation=self.args.min_saturation,
            min_value=self.args.min_value,
            hue_high1=self.args.hue_high1,
            hue_low2=self.args.hue_low2,
            device=self.args.yolo_device or None,
        )

    def _update_detector_controls(self):
        yolo = self.detector_mode.currentData() == "yolo"
        self.model_path.setEnabled(yolo and not self.planning and not self.executing)
        self.target_class.setEnabled(yolo and not self.planning and not self.executing)
        self.yolo_confidence.setEnabled(yolo and not self.planning and not self.executing)

    def _detector_mode_changed(self):
        self._update_detector_controls()
        self._apply_detector_config()

    def _apply_detector_config(self):
        if self.planning or self.executing:
            return
        try:
            config = self._current_detector_config()
            self.detector = TargetDetector(config)
            self.detector_value.setText("YOLO追踪" if config.mode == "yolo" else "像素追踪")
        except Exception as exc:
            QMessageBox.warning(self, "识别参数无效", str(exc))
            return
        self.detection_generation += 1
        self.last_detection = Detection(False)
        self._invalidate_approved_trajectory_silent("识别参数已修改，请重新识别并规划")
        self._set_status(
            "YOLO 模型将在识别时加载" if config.mode == "yolo" else "已切换到像素追踪",
            "statusIdle",
        )

    def _set_detector_controls_enabled(self, enabled):
        self.detector_mode.setEnabled(enabled)
        self.apply_detector_button.setEnabled(enabled)
        self._update_detector_controls()

    def _set_auxiliary_controls_enabled(self, enabled):
        enabled = bool(enabled and not self.emergency_latched)
        self.grasp_button.setEnabled(enabled)
        self.home_button.setEnabled(enabled)
        self._update_auto_sequence_button(enabled)

    def _update_auto_sequence_button(self, controls_enabled=True):
        ready = bool(
            controls_enabled
            and not self.emergency_latched
            and not self.planning
            and not self.executing
            and not self.auto_sequence_active
            and self.recognition_active
            and self.last_detection.found
            and getattr(self.last_detection, "camera_point", None) is not None
        )
        self.auto_sequence_button.setEnabled(ready)

    def _set_status(self, text, object_name):
        self.status.setText(text)
        self.status.setObjectName(object_name)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def _start_detection(self, image, depth, matrix, stamp):
        if self.detection_busy:
            return
        self.detection_busy = True
        generation = self.detection_generation
        detector = self.detector
        threading.Thread(
            target=self._detection_worker,
            args=(detector, image.copy(), None if depth is None else depth.copy(), matrix, stamp, generation),
            daemon=True,
        ).start()

    def _detection_worker(self, detector, image, depth, matrix, stamp, generation):
        try:
            detection = detector.detect(image)
            detection.depth = self._depth_for_detection(depth, detection)
            detection.stamp = stamp
            detection.generation = generation
            detection.detector_mode = detector.config.mode
            detection.class_names = detector.class_names() if detector.config.mode == "yolo" else []
            detection.camera_point = None
            if detection.found and detection.depth > 0.0 and matrix is not None:
                fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
                cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
                detection.camera_point = np.array([
                    (detection.u - cx) * detection.depth / fx,
                    (detection.v - cy) * detection.depth / fy,
                    detection.depth,
                    1.0,
                ], dtype=np.float64)
            self.worker_signals.detection_finished.emit(detection, "")
        except Exception as exc:
            self.worker_signals.detection_finished.emit(None, str(exc))

    def _depth_for_detection(self, depth, detection):
        if not detection.found or depth is None:
            return 0.0
        x, y = int(round(detection.u)), int(round(detection.v))
        radius = self.args.depth_window_px
        y0, y1 = max(0, y - radius), min(depth.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(depth.shape[1], x + radius + 1)
        roi = depth[y0:y1, x0:x1]
        if detection.mask is not None and detection.mask.shape == depth.shape:
            values = roi[detection.mask[y0:y1, x0:x1] > 0]
        else:
            values = roi.reshape(-1)
        values = values[np.isfinite(values)]
        values = values[(values > self.args.min_depth_m) & (values < self.args.max_depth_m)]
        return float(np.median(values)) if values.size else 0.0

    def _detection_finished(self, detection, error):
        self.detection_busy = False
        if error:
            self.last_detection = Detection(False)
            self.last_detection.reason = error
            self._set_status(f"识别失败: {error}", "statusMissing")
            return
        if detection.generation != self.detection_generation:
            return
        self.last_detection = detection
        if detection.class_names:
            current = self.target_class.currentText()
            existing = [self.target_class.itemText(i) for i in range(self.target_class.count())]
            if existing != detection.class_names:
                self.target_class.blockSignals(True)
                self.target_class.clear()
                self.target_class.addItems(detection.class_names)
                self.target_class.setCurrentText(current)
                self.target_class.blockSignals(False)
        if detection.found and detection.camera_point is None:
            detection.reason = "目标缺少有效深度或相机内参"
        self._update_auto_sequence_button()

    def _update_frame(self):
        if not rclpy.ok():
            self.timer.stop()
            return
        self.executor.spin_once(timeout_sec=0.0)
        image, depth, matrix, stamp = self.node.snapshot()
        if image is None:
            publisher_count = self.node.count_publishers(self.args.image_topic)
            if publisher_count == 0 and time.monotonic() - self.started_at > 1.5:
                self.connection_label.setText("RealSense 未启动")
                self.video.setText(
                    "未收到彩色图像\n\n图像话题没有发布者，请先启动 RealSense 相机"
                )
                self._set_status("相机未启动", "statusMissing")
            else:
                self.connection_label.setText("等待相机首帧")
            return
        self.connection_label.setText("相机已连接")
        overlay = image.copy()
        if self.recognition_active:
            self._start_detection(image, depth, matrix, stamp)
            detection = self.last_detection
            if detection.found:
                if detection.contour is not None:
                    cv2.drawContours(overlay, [detection.contour], -1, (0, 220, 80), 2)
                elif detection.bbox is not None:
                    x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 80), 2)
                center = (int(round(detection.u)), int(round(detection.v)))
                cv2.drawMarker(overlay, center, (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
                mode_label = "YOLO" if detection.detector_mode == "yolo" else "PIXEL"
                label = f"{mode_label} | {(detection.label or 'TARGET').upper()}"
                if detection.confidence is not None:
                    label += f" {detection.confidence:.2f}"
                if detection.depth > 0.0:
                    label += f"  {detection.depth:.3f} m"
                cv2.putText(overlay, label, (max(8, center[0] - 80), max(24, center[1] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2, cv2.LINE_AA)
                if self.executing:
                    self._set_status("正在执行真机轨迹…", "statusBusy")
                elif self.planning:
                    self._set_status("正在规划…", "statusBusy")
                elif time.monotonic() >= self.plan_feedback_until:
                    self._set_status("已识别红色目标", "statusFound")
                self.center_value.setText(f"{detection.u:.1f}, {detection.v:.1f} px")
                self.area_value.setText(f"{detection.area:.0f} px²")
                self.depth_value.setText(f"{detection.depth:.3f} m" if detection.depth > 0.0 else "无有效深度")
                self.plan_button.setEnabled(
                    not self.planning and not self.executing and detection.camera_point is not None
                )
            else:
                reason = getattr(detection, "reason", "")
                cv2.putText(overlay, "NO TARGET", (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (40, 40, 230), 2, cv2.LINE_AA)
                if not self.executing:
                    self._set_status(f"未识别到目标{': ' + reason if reason else ''}", "statusMissing")
                self.center_value.setText("-")
                self.area_value.setText("-")
                self.depth_value.setText("-")
                self.plan_button.setEnabled(False)
        else:
            cv2.putText(overlay, "RECOGNITION PAUSED", (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (220, 220, 220), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video.setPixmap(pixmap)

    def _planner_args(self):
        detector = self.detector.config
        return SimpleNamespace(
            calibration=self.args.calibration,
            image_topic=self.args.image_topic,
            camera_info_topic=self.args.camera_info_topic,
            depth_topic=self.args.depth_topic,
            target_distance=self.args.target_distance,
            base_frame=self.args.base_frame,
            gripper_frame=self.args.gripper_frame,
            end_effector_link="gripperStator",
            group="z1_arm",
            move_group_action="/move_action",
            trajectory_action="/joint_trajectory_controller/follow_joint_trajectory",
            mode="ray",
            approach_distance=self.args.approach_distance,
            step_size=0.0,
            z_offset=self.args.z_offset,
            servo_step_size=0.005,
            servo_gain=0.02,
            servo_deadband_px=35.0,
            servo_forward_step=0.0,
            goal_tolerance=0.008,
            orientation_tolerance=np.deg2rad(5.0),
            # Roll about tool +X is intentionally left free here. joint6 owns
            # that rotation through its independent 90 deg joint constraint.
            orientation_tolerances=(
                np.pi,
                np.deg2rad(self.args.grasp_pitch_tolerance_deg),
                np.deg2rad(5.0),
            ),
            orientation_weight=1.0,
            target_orientation_quat=None,
            target_joint6=np.deg2rad(self.args.joint6_target_deg),
            target_joint6_tolerance=np.deg2rad(self.args.joint6_tolerance_deg),
            max_joint_delta=0.12,
            local_joint_limit=0.0,
            speed_scale=0.05,
            planning_time=8.0,
            planning_attempts=5,
            action_wait_timeout=15.0,
            wait_timeout=10.0,
            detect_timeout=10.0,
            allow_latest_tf_fallback=True,
            detect_only=False,
            execute=False,
            no_preview=True,
            detector=detector.mode,
            yolo=detector.mode == "yolo",
            yolo_model=detector.model_path,
            yolo_class=detector.target_class,
            yolo_conf=detector.confidence,
            yolo_device=detector.device or "",
            min_area_px=self.args.min_area_px,
            min_saturation=self.args.min_saturation,
            min_value=self.args.min_value,
            hue_high1=self.args.hue_high1,
            hue_low2=self.args.hue_low2,
            depth_window_px=self.args.depth_window_px,
            min_depth_m=self.args.min_depth_m,
            max_depth_m=self.args.max_depth_m,
            wait_for_trigger=False,
            trigger_service="/red_target/start",
        )

    def _plan(self):
        if (
            self.emergency_latched
            or self.planning
            or self.executing
            or not self.last_detection.found
            or self.last_detection.camera_point is None
        ):
            return
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.approved_action = None
        self.execute_button.setEnabled(False)
        self.planning = True
        self._planning_action = "approach"
        self._planning_safety_generation = self.safety_generation
        self._planning_detection = copy.deepcopy(self.last_detection)
        self._set_detector_controls_enabled(False)
        self._set_auxiliary_controls_enabled(False)
        self.plan_button.setEnabled(False)
        self._set_status("正在规划…", "statusBusy")
        threading.Thread(target=self._planning_worker, daemon=True).start()

    def _start_auto_sequence(self):
        if (
            self.emergency_latched
            or self.planning
            or self.executing
            or self.auto_sequence_active
            or not self.recognition_active
            or not self.last_detection.found
            or self.last_detection.camera_point is None
        ):
            return
        answer = QMessageBox.question(
            self,
            "确认一键抓取放置",
            "将自动执行以下真实机械臂流程：\n\n"
            "1. 接近目标并张开夹爪\n"
            f"2. 夹爪闭合至 {self.args.gripper_close_deg:.1f}°\n"
            "3. 六轴返回 HOME（全部 0 rad）\n"
            f"4. 在 HOME 打开夹爪至 {self.args.gripper_open_rad:.2f} rad\n\n"
            "流程启动后各阶段会自动规划并执行。请确认全程无碰撞、"
            "HOME 位置适合放置物体，且实体急停可用。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.auto_sequence_active = True
        self.auto_sequence_button.setEnabled(False)
        self._set_status("一键流程：正在规划接近…", "statusBusy")
        self._plan()

    def _planning_worker(self):
        planner = None
        try:
            args = self._planner_args()
            planner = RedTargetHandeyeApproach(args)
            planner.wait_for_planning_inputs()
            frozen_detection = self._planning_detection
            camera_point = frozen_detection.camera_point.copy()
            stamp = frozen_detection.stamp
            (
                target_base,
                approach_base,
                motion_target,
                _,
                gripper_base,
                _,
            ) = planner.compute_approach_point(camera_point, stamp)
            planner.args.target_orientation_quat = horizontal_grasp_orientation(
                target_base - gripper_base,
                self.args.grasp_pitch_deg,
            )
            motion_target = offset_along_base_y(
                motion_target,
                self.args.y_offset,
            )
            motion_target = offset_along_tool_x(
                motion_target,
                planner.args.target_orientation_quat,
                self.args.insertion_offset_m,
            )
            target_radius = float(np.linalg.norm(target_base))
            approach_radius = float(np.linalg.norm(approach_base))
            cartesian_distance = float(np.linalg.norm(motion_target - gripper_base))
            best = None
            errors = []
            for _ in range(5):
                try:
                    candidate = planner.plan_move_group_goal(motion_target)
                except RuntimeError as exc:
                    errors.append(str(exc))
                    continue
                trajectory = candidate.planned_trajectory.joint_trajectory
                max_delta, _ = planner.trajectory_joint_delta(trajectory)
                if best is None or max_delta < best[0]:
                    best = (max_delta, candidate)
            if best is None:
                raise RuntimeError(
                    "完整接近目标不可达："
                    f"目标距基座 {target_radius:.3f} m，"
                    f"接近点距基座 {approach_radius:.3f} m，"
                    f"末端需移动 {cartesian_distance:.3f} m。"
                    f"MoveIt: {errors[-1] if errors else 'no solution'}"
                )
            max_delta, result = best
            trajectory = result.planned_trajectory.joint_trajectory
            if "joint6" not in trajectory.joint_names:
                raise RuntimeError("规划结果缺少 joint6")
            joint6_index = trajectory.joint_names.index("joint6")
            joint6_final = float(trajectory.points[-1].positions[joint6_index])
            joint6_target = np.deg2rad(self.args.joint6_target_deg)
            joint6_tolerance = np.deg2rad(self.args.joint6_tolerance_deg)
            if abs(joint6_final - joint6_target) > joint6_tolerance:
                raise RuntimeError(
                    f"joint6 终点 {np.rad2deg(joint6_final):.1f}° 未达到目标 "
                    f"{self.args.joint6_target_deg:.1f}°"
                )
            gripper_start = self.node.joint_position(self.args.gripper_joint)
            if gripper_start is None:
                raise RuntimeError(f"未收到夹爪关节 {self.args.gripper_joint} 的状态")
            add_gripper_motion(
                result.planned_trajectory,
                gripper_start,
                self.args.gripper_open_rad,
                self.args.gripper_joint,
            )
            planner.publish_display_trajectory(result)
            rclpy.spin_once(planner, timeout_sec=0.2)
            message = (
                "夹爪接近轨迹规划成功："
                f"{len(trajectory.points)} 个轨迹点，"
                f"末端移动 {cartesian_distance:.3f} m，"
                f"最大关节变化 {max_delta:.3f} rad；"
                f"joint6→{self.args.joint6_target_deg:.1f}°，"
                f"水平插入 pitch {self.args.grasp_pitch_deg:.1f}°"
                f"±{self.args.grasp_pitch_tolerance_deg:.1f}°，"
                f"基座 +Y 偏移 {self.args.y_offset:.3f} m，"
                f"沿夹爪 +X 深入 {self.args.insertion_offset_m:.3f} m，"
                f"夹爪→OPEN ({self.args.gripper_open_rad:.2f} rad)"
            )
            approved = copy.deepcopy(result.planned_trajectory)
            self.worker_signals.plan_finished.emit(True, message, approved)
        except Exception as exc:
            self.worker_signals.plan_finished.emit(False, f"规划失败：{exc}", None)
        finally:
            if planner is not None:
                planner.destroy_node()

    def _plan_grasp(self):
        self._start_auxiliary_plan("grasp")

    def _plan_home(self):
        self._start_auxiliary_plan("home")

    def _start_auxiliary_plan(self, action):
        if self.emergency_latched or self.planning or self.executing:
            return
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.approved_action = None
        self.execute_button.setEnabled(False)
        self.planning = True
        self._planning_action = action
        self._planning_safety_generation = self.safety_generation
        self._planning_detection = None
        self._set_detector_controls_enabled(False)
        self._set_auxiliary_controls_enabled(False)
        self.plan_button.setEnabled(False)
        label = {
            "grasp": "夹爪闭合",
            "home": "返回 HOME",
            "release": "夹爪释放",
        }.get(action, action)
        self._set_status(f"正在规划{label}轨迹…", "statusBusy")
        threading.Thread(
            target=self._auxiliary_planning_worker, args=(action,), daemon=True
        ).start()

    def _auxiliary_planning_worker(self, action):
        planner = None
        try:
            args = self._planner_args()
            if action in ("grasp", "release"):
                args.group = "z1_gripper"
                args.trajectory_action = (
                    "/gripper_trajectory_controller/follow_joint_trajectory"
                )
                args.speed_scale = 0.10
                args.target_joint6 = None
                planner = RedTargetHandeyeApproach(args)
                target_gripper = (
                    gripper_close_target_rad(self.args.gripper_close_deg)
                    if action == "grasp"
                    else self.args.gripper_open_rad
                )
                current_gripper = self.node.joint_position(self.args.gripper_joint)
                if current_gripper is None:
                    raise RuntimeError(
                        f"未收到夹爪关节 {self.args.gripper_joint} 的状态"
                    )
                if action == "grasp" and target_gripper < current_gripper - 0.01:
                    raise RuntimeError(
                        f"当前夹爪已闭合到 {np.rad2deg(current_gripper):.1f}°；"
                        f"执行 {self.args.gripper_close_deg:.1f}°目标反而会张开，已拒绝"
                    )
                result = planner.plan_move_group_joint_goal(
                    {self.args.gripper_joint: target_gripper}
                )
                trajectory = result.planned_trajectory.joint_trajectory
                if action == "grasp":
                    message = (
                        f"夹爪闭合轨迹规划成功：当前 joint={np.rad2deg(current_gripper):.1f}°，"
                        f"目标 joint={np.rad2deg(target_gripper):.1f}°"
                        f"（闭合角 {self.args.gripper_close_deg:.1f}°），"
                        f"(joint={target_gripper:.3f} rad)，{len(trajectory.points)} 个轨迹点"
                    )
                else:
                    message = (
                        f"HOME 放置释放规划成功：夹爪→OPEN "
                        f"({target_gripper:.3f} rad)，{len(trajectory.points)} 个轨迹点"
                    )
            elif action == "home":
                args.group = "z1_arm"
                args.target_joint6 = None
                args.orientation_weight = 0.0
                args.speed_scale = 0.05
                planner = RedTargetHandeyeApproach(args)
                result = planner.plan_move_group_joint_goal(HOME_JOINT_TARGETS)
                trajectory = result.planned_trajectory.joint_trajectory
                if not trajectory.joint_names or not trajectory.points:
                    raise RuntimeError("MoveIt 返回了空的 HOME 轨迹")
                final = trajectory.points[-1]
                final_by_joint = dict(zip(trajectory.joint_names, final.positions))
                missing = [name for name in HOME_JOINT_TARGETS if name not in final_by_joint]
                if missing:
                    raise RuntimeError(f"HOME 轨迹缺少关节：{', '.join(missing)}")
                max_home_error = max(
                    abs(final_by_joint[name] - target)
                    for name, target in HOME_JOINT_TARGETS.items()
                )
                if max_home_error > 0.01:
                    raise RuntimeError(
                        f"HOME 轨迹终点误差 {max_home_error:.4f} rad 超过 0.01 rad"
                    )
                message = (
                    f"返回 HOME 轨迹规划成功：六轴目标均为 0 rad，"
                    f"终点最大误差 {max_home_error:.4f} rad，"
                    f"{len(trajectory.points)} 个轨迹点"
                )
            else:
                raise RuntimeError(f"未知辅助动作：{action}")

            if not trajectory.joint_names or not trajectory.points:
                raise RuntimeError("MoveIt 返回了空轨迹")
            planner.publish_display_trajectory(result)
            rclpy.spin_once(planner, timeout_sec=0.2)
            approved = copy.deepcopy(result.planned_trajectory)
            self.worker_signals.plan_finished.emit(True, message, approved)
        except Exception as exc:
            self.worker_signals.plan_finished.emit(False, f"规划失败：{exc}", None)
        finally:
            if planner is not None:
                planner.destroy_node()

    def _planning_finished(self, success, message, trajectory):
        self.planning = False
        if (
            self.emergency_latched
            or self._planning_safety_generation != self.safety_generation
        ):
            self.auto_sequence_active = False
            self.approved_trajectory = None
            self.execute_button.setEnabled(False)
            self.reset_emergency_button.setEnabled(self.emergency_latched)
            self._set_status(
                "急停已锁止" if self.emergency_latched else "急停后请重新规划",
                "statusMissing",
            )
            return
        controls_enabled = not self.executing and not self.auto_sequence_active
        self._set_detector_controls_enabled(controls_enabled)
        self._set_auxiliary_controls_enabled(controls_enabled)
        if success:
            self.approved_trajectory = trajectory
            self.approved_summary = message
            self.approved_action = self._planning_action
            self.approved_detection = (
                copy.deepcopy(self._planning_detection)
                if self._planning_action == "approach"
                else None
            )
            self._set_status("轨迹已发布到 RViz", "statusFound")
            self.plan_feedback_until = time.monotonic() + 8.0
            self.execute_button.setEnabled(not self.auto_sequence_active)
            if self.auto_sequence_active:
                QTimer.singleShot(0, self._execute_auto_approved_trajectory)
        else:
            self.auto_sequence_active = False
            self.approved_trajectory = None
            self.approved_summary = ""
            self.approved_detection = None
            self.execute_button.setEnabled(False)
            self._set_status("规划失败", "statusMissing")
            QMessageBox.warning(self, "规划失败", message)
            self._set_detector_controls_enabled(True)
            self._set_auxiliary_controls_enabled(True)
        self.plan_button.setEnabled(
            not self.emergency_latched
            and self.recognition_active
            and self.last_detection.found
            and not self.executing
        )
        self.statusBar().showMessage(message, 12000)

    def _approach_target_validation_error(self):
        if self.approved_action != "approach":
            return ""
        if not self.last_detection.found or self.approved_detection is None:
            return "红色目标已离开画面，请重新识别并规划"
        pixel_shift = float(np.hypot(
            self.last_detection.u - self.approved_detection.u,
            self.last_detection.v - self.approved_detection.v,
        ))
        depth_shift = abs(self.last_detection.depth - self.approved_detection.depth)
        if pixel_shift > 35.0 or (
            self.last_detection.depth > 0.0
            and self.approved_detection.depth > 0.0
            and depth_shift > 0.05
        ):
            return (
                f"红色目标已移动（像素偏移 {pixel_shift:.1f} px，"
                f"深度变化 {depth_shift:.3f} m），请重新规划"
            )
        return ""

    def _execute_auto_approved_trajectory(self):
        if not self.auto_sequence_active or self.emergency_latched:
            return
        error = self._approach_target_validation_error()
        if error:
            self.auto_sequence_active = False
            self._invalidate_approved_trajectory(error)
            self._set_auxiliary_controls_enabled(True)
            return
        if self.approved_trajectory is None:
            self.auto_sequence_active = False
            self._set_status("一键流程中断：没有可执行轨迹", "statusMissing")
            return
        trajectory = copy.deepcopy(self.approved_trajectory)
        self._begin_execution(trajectory, self.approved_action)

    def _confirm_execute(self):
        if self.emergency_latched or self.executing or self.planning or self.approved_trajectory is None:
            return
        error = self._approach_target_validation_error()
        if error:
            self._invalidate_approved_trajectory(error)
            return
        action_note = {
            "approach": (
                f"执行时 joint6 将转至 {self.args.joint6_target_deg:.1f}°，"
                f"夹爪将同步打开至 OPEN ({self.args.gripper_open_rad:.2f} rad)。"
            ),
            "grasp": f"夹爪将闭合至 {self.args.gripper_close_deg:.1f}°。",
            "home": "机械臂六个关节将返回 0 rad HOME 位。",
            "release": f"夹爪将在 HOME 打开至 {self.args.gripper_open_rad:.2f} rad。",
        }.get(self.approved_action, "")
        answer = QMessageBox.question(
            self,
            "确认执行真机轨迹",
            "即将执行刚才在 RViz 中显示的同一条完整轨迹。\n\n"
            f"{self.approved_summary}\n\n"
            f"{action_note}\n"
            "请确认轨迹全程无碰撞、工作区无人，且实体急停可用。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        trajectory = copy.deepcopy(self.approved_trajectory)
        self._begin_execution(trajectory, self.approved_action)

    def _begin_execution(self, trajectory, action):
        self.executing = True
        self._set_detector_controls_enabled(False)
        self._set_auxiliary_controls_enabled(False)
        self.plan_button.setEnabled(False)
        self.execute_button.setEnabled(False)
        self._set_status("正在验证轨迹起点…", "statusBusy")
        threading.Thread(
            target=self._execution_worker, args=(trajectory, action), daemon=True
        ).start()

    def _execution_worker(self, trajectory, action):
        executor_node = None
        try:
            executor_node = ApprovedTrajectoryExecutor(self.args.execute_action)
            with self.active_executor_lock:
                self.active_executor = executor_node
            if self.emergency_latched:
                executor_node.request_stop()
            try:
                start_error = executor_node.execute(
                    trajectory, self.args.execution_start_tolerance
                )
            except TrajectoryExecutionError as exc:
                if action != "home" or exc.error_code != MoveItErrorCodes.TIMED_OUT:
                    raise
                worst_position, worst_velocity = executor_node.wait_for_settled_goal(
                    HOME_JOINT_TARGETS,
                    HOME_POSITION_TOLERANCE_RAD,
                    HOME_VELOCITY_TOLERANCE_RAD_S,
                    HOME_SETTLE_TIMEOUT_S,
                    HOME_SETTLE_SAMPLES,
                )
                self.worker_signals.execution_finished.emit(
                    True,
                    "MoveIt 的 HOME 执行计时已到，但机械臂已安全稳定在 HOME："
                    f"最大位置误差 {worst_position[1]:.4f} rad "
                    f"({worst_position[0]})，最大速度 {worst_velocity[1]:.4f} rad/s "
                    f"({worst_velocity[0]})",
                )
                return
            self.worker_signals.execution_finished.emit(
                True, f"轨迹执行成功；执行前起点误差 {start_error:.4f} rad"
            )
        except Exception as exc:
            self.worker_signals.execution_finished.emit(False, f"轨迹执行失败：{exc}")
        finally:
            with self.active_executor_lock:
                self.active_executor = None
            if executor_node is not None:
                executor_node.destroy_node()

    def _execution_finished(self, success, message):
        completed_action = self.approved_action
        continue_auto = bool(
            success and self.auto_sequence_active and not self.emergency_latched
        )
        self.executing = False
        self.reset_emergency_button.setEnabled(self.emergency_latched)
        self._set_detector_controls_enabled(not continue_auto)
        self._set_auxiliary_controls_enabled(not continue_auto)
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.approved_action = None
        self.execute_button.setEnabled(False)
        if self.emergency_latched:
            self.auto_sequence_active = False
            self._set_status("急停已锁止", "statusMissing")
        elif continue_auto:
            next_action = AUTO_SEQUENCE_NEXT_ACTION.get(completed_action)
            if next_action is None:
                self.auto_sequence_active = False
                self._set_detector_controls_enabled(True)
                self._set_auxiliary_controls_enabled(True)
                self._set_status("一键抓取放置完成", "statusFound")
                self.statusBar().showMessage(
                    "接近、抓取、返回 HOME 和释放均已完成", 15000
                )
                return
            label = {
                "grasp": "夹爪闭合",
                "home": "返回 HOME",
                "release": "HOME 放置释放",
            }[next_action]
            self._set_status(f"一键流程：正在规划{label}…", "statusBusy")
            QTimer.singleShot(
                0, lambda action=next_action: self._start_auxiliary_plan(action)
            )
            return
        elif success:
            self._set_status("轨迹执行完成", "statusFound")
        else:
            was_auto = self.auto_sequence_active
            self.auto_sequence_active = False
            self._set_status("轨迹执行失败", "statusMissing")
            QMessageBox.warning(
                self,
                "一键流程已中断" if was_auto else "轨迹执行失败",
                message,
            )
        self.plan_button.setEnabled(
            not self.emergency_latched and self.recognition_active and self.last_detection.found
        )
        self.statusBar().showMessage(message, 15000)
        self._update_auto_sequence_button()

    def _invalidate_approved_trajectory(self, message):
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.approved_action = None
        self.execute_button.setEnabled(False)
        self._set_status("审核轨迹已失效", "statusMissing")
        self.statusBar().showMessage(message, 15000)
        QMessageBox.warning(self, "审核轨迹已失效", message)

    def _invalidate_approved_trajectory_silent(self, message):
        if self.approved_trajectory is None:
            return
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.approved_action = None
        self.execute_button.setEnabled(False)
        self.statusBar().showMessage(message, 10000)

    def closeEvent(self, event):
        if self.executing or self.planning or self.auto_sequence_active:
            QMessageBox.warning(
                self,
                "任务正在运行",
                "规划、轨迹执行或一键流程运行期间不能关闭控制窗口；请先急停。",
            )
            event.ignore()
            return
        self.timer.stop()
        self.executor.remove_node(self.node)
        self.executor.shutdown(timeout_sec=0.5)
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(description="Live red-target recognition GUI for the Z1 wrist camera.")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--base-frame", default="link00")
    parser.add_argument("--gripper-frame", default="gripperStator")
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument(
        "--z-offset",
        type=float,
        default=DEFAULT_BOTTLE_Z_OFFSET_M,
        help="Additional base-frame Z offset for the visual approach target.",
    )
    parser.add_argument(
        "--y-offset",
        type=float,
        default=DEFAULT_BOTTLE_Y_OFFSET_M,
        help="Final link00/base-frame Y offset for the approach target.",
    )
    parser.add_argument("--execute-action", default="/execute_trajectory")
    parser.add_argument("--execution-start-tolerance", type=float, default=0.03)
    parser.add_argument("--joint6-target-deg", type=float, default=90.0)
    parser.add_argument("--joint6-tolerance-deg", type=float, default=2.0)
    parser.add_argument(
        "--grasp-pitch-deg", type=float, default=DEFAULT_GRASP_PITCH_DEG
    )
    parser.add_argument(
        "--grasp-pitch-tolerance-deg",
        type=float,
        default=GRASP_PITCH_TOLERANCE_DEG,
    )
    parser.add_argument(
        "--insertion-offset-m", type=float, default=DEFAULT_INSERTION_OFFSET_M
    )
    parser.add_argument("--gripper-joint", default="jointGripper")
    parser.add_argument(
        "--gripper-open-rad",
        type=float,
        default=DEFAULT_GRIPPER_OPEN_RAD,
        help="Named OPEN target for jointGripper (default: -0.90 rad)",
    )
    parser.add_argument(
        "--gripper-close-deg",
        type=float,
        default=DEFAULT_GRIPPER_CLOSE_DEG,
        help="Positive closing angle shown to the user; maps to a negative Z1 joint angle.",
    )
    parser.add_argument("--target-distance", type=float, default=0.45)
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
    if not Path(args.calibration).exists():
        raise FileNotFoundError(f"Calibration file not found: {args.calibration}")
    if not -160.0 <= args.joint6_target_deg <= 160.0:
        raise ValueError("joint6 目标角必须在 [-160°, 160°] 内")
    if not -np.pi / 2.0 <= args.gripper_open_rad <= 0.0:
        raise ValueError("夹爪 OPEN 目标必须在 [-pi/2, 0] rad 内")
    if args.joint6_tolerance_deg <= 0.0:
        raise ValueError("joint6 容差必须大于 0°")
    if not -45.0 <= args.grasp_pitch_deg <= 45.0:
        raise ValueError("水平抓取 pitch 必须在 [-45°, 45°] 内")
    if not 0.1 <= args.grasp_pitch_tolerance_deg <= 10.0:
        raise ValueError("水平抓取 pitch 容差必须在 [0.1°, 10°] 内")
    if not 0.0 <= args.insertion_offset_m <= 0.15:
        raise ValueError("夹爪 +X 插入深入量必须在 [0, 0.15] m 内")
    if not -0.15 <= args.y_offset <= 0.15:
        raise ValueError("基座 Y 轴偏移必须在 [-0.15, 0.15] m 内")
    gripper_close_target_rad(args.gripper_close_deg)
    return args


def main():
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = CameraNode(args)
    app = QApplication(remove_ros_args(args=sys.argv))
    app.setStyle("Fusion")
    window = RedTargetWindow(args, node)
    window.show()
    result = app.exec_()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    raise SystemExit(result)


if __name__ == "__main__":
    main()
