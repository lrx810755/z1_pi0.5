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

from z1_examples.red_target_handeye_approach import RedTargetHandeyeApproach
from z1_examples.target_detector import DetectorConfig, TargetDetector, add_detector_arguments


DEFAULT_CALIBRATION = Path.home() / "ros2_ws" / "z1_handeye_charuco_gripper_to_camera.yaml"


@dataclass
class Detection:
    found: bool
    u: float = 0.0
    v: float = 0.0
    area: float = 0.0
    depth: float = 0.0


class CameraNode(Node):
    def __init__(self, args):
        super().__init__("red_target_gui")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.image = None
        self.depth = None
        self.camera_matrix = None
        self.image_stamp = None
        self.create_subscription(Image, args.image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(Image, args.depth_topic, self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._info_cb, 10)

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
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.client = ActionClient(self, ExecuteTrajectory, action_name)

    def _joint_state_cb(self, msg):
        with self.lock:
            self.joint_positions = {
                name: float(position) for name, position in zip(msg.name, msg.position)
            }

    def validate_start(self, robot_trajectory, tolerance, timeout=5.0):
        trajectory = robot_trajectory.joint_trajectory
        if not trajectory.joint_names or not trajectory.points:
            raise RuntimeError("已审核轨迹为空")
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
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
        if not self.client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveIt /execute_trajectory action 不可用")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_trajectory
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("MoveIt 拒绝执行已审核轨迹")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise RuntimeError("MoveIt 执行没有返回结果")
        code = wrapped_result.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt 轨迹执行失败，error_code={code}")
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
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.plan_feedback_until = 0.0
        self.started_at = time.monotonic()
        self.worker_signals = WorkerSignals()
        self.worker_signals.plan_finished.connect(self._planning_finished)
        self.worker_signals.execution_finished.connect(self._execution_finished)
        self.worker_signals.detection_finished.connect(self._detection_finished)
        self.setWindowTitle("Z1 红色目标识别")
        self.setMinimumSize(900, 650)
        self.resize(1080, 760)
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
        self.video.setMinimumSize(640, 480)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setObjectName("video")
        content.addWidget(self.video, 1)

        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(300)
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
        self.detector_mode.addItem("像素追踪", "pixel")
        self.detector_mode.addItem("YOLO追踪", "yolo")
        initial_mode = "yolo" if getattr(self.args, "yolo", False) else self.args.detector
        self.detector_mode.setCurrentIndex(1 if initial_mode == "yolo" else 0)
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

        details = QFormLayout()
        details.setVerticalSpacing(10)
        self.center_value = QLabel("-")
        self.area_value = QLabel("-")
        self.depth_value = QLabel("-")
        self.detector_value = QLabel("像素追踪" if initial_mode == "pixel" else "YOLO追踪")
        self.mode_value = QLabel("完整接近")
        details.addRow("目标中心", self.center_value)
        details.addRow("轮廓面积", self.area_value)
        details.addRow("目标深度", self.depth_value)
        details.addRow("实际识别", self.detector_value)
        details.addRow("规划模式", self.mode_value)
        side_layout.addLayout(details)
        side_layout.addStretch()

        note = QLabel("先在 RViz 审核完整轨迹，再执行同一条缓存轨迹。机械臂位置变化后必须重新规划。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        side_layout.addWidget(note)
        content.addWidget(side)
        root.addLayout(content, 1)
        self.setCentralWidget(central)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f4f6f8; color: #18212b; font-size: 14px; }
            QLabel#title { font-size: 22px; font-weight: 600; }
            QLabel#muted { color: #65717d; }
            QLabel#video { background: #101419; color: #aab3bc; border: 1px solid #c8d0d8; }
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
        """)

    def _toggle_recognition(self):
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
            end_effector_link=self.args.end_effector_link,
            group="z1_arm",
            move_group_action="/move_action",
            trajectory_action="/joint_trajectory_controller/follow_joint_trajectory",
            mode="ray",
            approach_distance=self.args.approach_distance,
            step_size=0.0,
            z_offset=0.0,
            servo_step_size=0.005,
            servo_gain=0.02,
            servo_deadband_px=35.0,
            servo_forward_step=0.0,
            goal_tolerance=0.008,
            orientation_tolerance=0.08,
            orientation_weight=1.0,
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
            self.planning
            or self.executing
            or not self.last_detection.found
            or self.last_detection.camera_point is None
        ):
            return
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.execute_button.setEnabled(False)
        self.planning = True
        self._planning_detection = copy.deepcopy(self.last_detection)
        self._set_detector_controls_enabled(False)
        self.plan_button.setEnabled(False)
        self._set_status("正在规划…", "statusBusy")
        threading.Thread(target=self._planning_worker, daemon=True).start()

    def _planning_worker(self):
        planner = None
        try:
            args = self._planner_args()
            planner = RedTargetHandeyeApproach(args)
            planner.wait_for_inputs()
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
            planner.publish_display_trajectory(result)
            rclpy.spin_once(planner, timeout_sec=0.2)
            trajectory = result.planned_trajectory.joint_trajectory
            message = (
                f"完整规划成功：{len(trajectory.points)} 个轨迹点，"
                f"末端移动 {cartesian_distance:.3f} m，"
                f"最大关节变化 {max_delta:.3f} rad"
            )
            approved = copy.deepcopy(result.planned_trajectory)
            self.worker_signals.plan_finished.emit(True, message, approved)
        except Exception as exc:
            self.worker_signals.plan_finished.emit(False, f"规划失败：{exc}", None)
        finally:
            if planner is not None:
                planner.destroy_node()

    def _planning_finished(self, success, message, trajectory):
        self.planning = False
        self._set_detector_controls_enabled(not self.executing)
        if success:
            self.approved_trajectory = trajectory
            self.approved_summary = message
            self.approved_detection = copy.deepcopy(self._planning_detection)
            self._set_status("轨迹已发布到 RViz", "statusFound")
            self.plan_feedback_until = time.monotonic() + 8.0
            self.execute_button.setEnabled(True)
        else:
            self.approved_trajectory = None
            self.approved_summary = ""
            self.approved_detection = None
            self.execute_button.setEnabled(False)
            self._set_status("规划失败", "statusMissing")
            QMessageBox.warning(self, "规划失败", message)
        self.plan_button.setEnabled(
            self.recognition_active and self.last_detection.found and not self.executing
        )
        self.statusBar().showMessage(message, 12000)

    def _confirm_execute(self):
        if self.executing or self.planning or self.approved_trajectory is None:
            return
        if not self.last_detection.found or self.approved_detection is None:
            self._invalidate_approved_trajectory("红色目标已离开画面，请重新识别并规划")
            return
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
            self._invalidate_approved_trajectory(
                f"红色目标已移动（像素偏移 {pixel_shift:.1f} px，深度变化 {depth_shift:.3f} m），请重新规划"
            )
            return
        answer = QMessageBox.question(
            self,
            "确认执行真机轨迹",
            "即将执行刚才在 RViz 中显示的同一条完整轨迹。\n\n"
            f"{self.approved_summary}\n\n"
            "执行会真实移动 Z1。请确认轨迹全程无碰撞、工作区无人，且急停可用。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        trajectory = copy.deepcopy(self.approved_trajectory)
        self.executing = True
        self._set_detector_controls_enabled(False)
        self.plan_button.setEnabled(False)
        self.execute_button.setEnabled(False)
        self._set_status("正在验证轨迹起点…", "statusBusy")
        threading.Thread(
            target=self._execution_worker, args=(trajectory,), daemon=True
        ).start()

    def _execution_worker(self, trajectory):
        executor_node = None
        try:
            executor_node = ApprovedTrajectoryExecutor(self.args.execute_action)
            start_error = executor_node.execute(
                trajectory, self.args.execution_start_tolerance
            )
            self.worker_signals.execution_finished.emit(
                True, f"轨迹执行成功；执行前起点误差 {start_error:.4f} rad"
            )
        except Exception as exc:
            self.worker_signals.execution_finished.emit(False, f"轨迹执行失败：{exc}")
        finally:
            if executor_node is not None:
                executor_node.destroy_node()

    def _execution_finished(self, success, message):
        self.executing = False
        self._set_detector_controls_enabled(True)
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
        self.execute_button.setEnabled(False)
        if success:
            self._set_status("轨迹执行完成", "statusFound")
        else:
            self._set_status("轨迹执行失败", "statusMissing")
            QMessageBox.warning(self, "轨迹执行失败", message)
        self.plan_button.setEnabled(
            self.recognition_active and self.last_detection.found
        )
        self.statusBar().showMessage(message, 15000)

    def _invalidate_approved_trajectory(self, message):
        self.approved_trajectory = None
        self.approved_summary = ""
        self.approved_detection = None
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
        self.execute_button.setEnabled(False)
        self.statusBar().showMessage(message, 10000)

    def closeEvent(self, event):
        if self.executing:
            QMessageBox.warning(self, "轨迹正在执行", "轨迹执行期间不能关闭控制窗口。")
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
    parser.add_argument("--end-effector-link", default="gripperStator")
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--execute-action", default="/execute_trajectory")
    parser.add_argument("--execution-start-tolerance", type=float, default=0.03)
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
