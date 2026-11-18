"""Record synchronized Z1 demonstrations into a recoverable raw dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from .paths import dataset_root
from .recording_sync import RecordingSynchronizer


JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "jointGripper",
)
TASK_INSTRUCTIONS = {
    "cup": "Physically grasp the cup, lift it clear, and place it in the box.",
    "apple": "Physically grasp the apple, lift it clear, and place it in the box.",
    "peach": "Physically grasp the peach, lift it clear, and place it in the box.",
    "mouse": "Physically grasp the mouse, lift it clear, and place it in the box.",
    "beverage_bottle": (
        "Physically grasp the bottle, lift it clear, and place it in the box."
    ),
}
FORMAT_VERSION = "z1_smolvla_raw_v1"


def new_episode_evidence() -> Dict[str, bool]:
    return {
        "physical_grasp_verified": False,
        "object_lifted_ever": False,
        "bilateral_contact_ever": False,
        "release_sequence_complete": False,
        "task_success": False,
    }


def update_episode_evidence(
    evidence: Dict[str, bool], status: Dict[str, Any]
) -> None:
    evidence["physical_grasp_verified"] |= bool(
        status.get("physical_grasp_ok", False)
    )
    evidence["object_lifted_ever"] |= bool(status.get("lifted", False))
    evidence["bilateral_contact_ever"] |= bool(
        status.get("bilateral_contact", False)
    )
    evidence["release_sequence_complete"] |= bool(
        status.get("release_sequence_complete", False)
    )
    evidence["task_success"] |= bool(status.get("task_success", False))


def episode_save_failures(
    evidence: Dict[str, bool],
    final_status: Dict[str, Any],
    require_task_success: bool,
) -> Sequence[str]:
    conditions = {
        "physical_grasp_never_verified": bool(
            evidence.get("physical_grasp_verified", False)
        ),
        "object_never_lifted_clear_of_table": bool(
            evidence.get("object_lifted_ever", False)
        ),
        "bilateral_contact_never_observed": bool(
            evidence.get("bilateral_contact_ever", False)
        ),
    }
    if require_task_success:
        conditions.update(
            {
                "release_sequence_not_completed": bool(
                    evidence.get("release_sequence_complete", False)
                    or final_status.get(
                        "release_sequence_complete", False
                    )
                ),
                "placement_not_stable_in_box": bool(
                    evidence.get("task_success", False)
                    or final_status.get("task_success", False)
                ),
            }
        )
    return [
        name for name, condition_passed in conditions.items()
        if not condition_passed
    ]


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def image_to_bgr(message: Image) -> np.ndarray:
    """Convert common ROS 8-bit image encodings without cv_bridge."""
    encoding = message.encoding.lower()
    channels = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {message.encoding!r}")
    row_bytes = int(message.width) * channels
    step = int(message.step) if message.step else row_bytes
    required = int(message.height) * step
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < required or step < row_bytes:
        raise ValueError(
            "invalid image buffer: "
            f"got {raw.size}, expected at least {required}"
        )
    rows = raw[:required].reshape(int(message.height), step)
    image = rows[:, :row_bytes].reshape(
        int(message.height), int(message.width), channels
    )
    if encoding in ("rgb8",):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding in ("rgba8",):
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding in ("bgra8", "8uc4"):
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


class EpisodeRecorder(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_episode_recorder")
        self.declare_parameter(
            "dataset_root",
            str(dataset_root("z1_tabletop_raw_left_wrist_v1")),
        )
        self.declare_parameter("fps", 10.0)
        self.declare_parameter("stream_timeout_s", 1.0)
        self.declare_parameter("sync_tolerance_s", 0.045)
        self.declare_parameter("sync_delay_s", 0.5)
        # The recorded action is the most recent command at or before the
        # sample time. Between two planned segments the command stream goes
        # silent while the controller keeps holding the previous target, so
        # the old bound of exactly one frame period left no margin and threw
        # away otherwise valid episodes. Keep the bound explicit and loose
        # enough to span a segment-planning pause; every frame also records
        # the real age of the command it used.
        self.declare_parameter("action_age_s", 0.40)
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("minimum_frames", 10)
        self.declare_parameter("require_task_success", True)
        self.declare_parameter(
            "wrist_rgb_topic", "/left_wrist_camera/color/image_raw"
        )
        self.declare_parameter(
            "vla_wrist_rgb_topic",
            "none",
        )
        self.declare_parameter(
            "global_rgb_topic", "/global_camera/color/image_raw"
        )
        self.declare_parameter("global_right_rgb_topic", "none")
        self.declare_parameter("camera_calibration_path", "")
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("action_topic", "/isaac_joint_commands")
        self.declare_parameter("task_status_topic", "/smolvla/status")

        self.dataset_root = Path(
            str(self.get_parameter("dataset_root").value)
        ).expanduser()
        self.fps = float(self.get_parameter("fps").value)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.stream_timeout = float(
            self.get_parameter("stream_timeout_s").value
        )
        self.jpeg_quality = int(
            self.get_parameter("jpeg_quality").value
        )
        self.minimum_frames = int(
            self.get_parameter("minimum_frames").value
        )
        self.require_task_success = bool(
            self.get_parameter("require_task_success").value
        )
        self.expected_image_shapes = {
            "wrist": (360, 640, 3),
            "global": (480, 640, 3),
        }
        self.vla_wrist_rgb_topic = str(
            self.get_parameter("vla_wrist_rgb_topic").value
        ).strip()
        if self.vla_wrist_rgb_topic.lower() == "none":
            self.vla_wrist_rgb_topic = ""
        if self.vla_wrist_rgb_topic:
            self.expected_image_shapes["vla_wrist"] = (360, 640, 3)
        self.global_right_rgb_topic = str(self.get_parameter("global_right_rgb_topic").value).strip()
        if self.global_right_rgb_topic.lower() == "none":
            self.global_right_rgb_topic = ""
        if self.global_right_rgb_topic:
            self.expected_image_shapes["global_right"] = (480, 640, 3)
        self.camera_names = tuple(self.expected_image_shapes)
        self.sync_tolerance = float(self.get_parameter("sync_tolerance_s").value)
        self.sync_delay = float(self.get_parameter("sync_delay_s").value)
        self.action_age = float(self.get_parameter("action_age_s").value)
        if not (np.isfinite(self.sync_tolerance) and np.isfinite(self.sync_delay)
                and 0 < self.sync_tolerance < 0.5 / self.fps
                and self.sync_tolerance < self.sync_delay <= 1.0):
            raise ValueError("sync requires 0 < tolerance < half a sample period and tolerance < delay <= 1")
        if not (np.isfinite(self.action_age) and 0 < self.action_age <= 1.0):
            raise ValueError("action_age_s must be finite and within (0, 1]")
        self._sync = RecordingSynchronizer(
            self.camera_names, tolerance=self.sync_tolerance,
            state_gap=1.0 / self.fps, action_age=self.action_age,
        )
        self._sync_failure = ""
        self._selected_sample = None
        self._last_sync_clock = None
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        if abs(self.fps - round(self.fps)) > 1.0e-6:
            raise ValueError("fps must be an integer value for LeRobot export")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        self._lock = threading.RLock()
        self._images: Dict[str, Tuple[Image, float, float]] = {}
        self._joint_state: Dict[str, float] = {}
        self._joint_state_time = 0.0
        self._joint_state_stamp = 0.0
        self._action: Dict[str, float] = {}
        self._action_time = 0.0
        self._action_stamp = 0.0
        self._task_status: Dict[str, Any] = {}
        self._task_status_time = 0.0
        self._randomization_status: Dict[str, Any] = {}
        self._episode_randomization: Dict[str, Any] = {}
        self._recording = False
        self._episode_index: Optional[int] = None
        self._episode_dir: Optional[Path] = None
        self._frames_file = None
        self._frame_count = 0
        self._dropped_frames = 0
        self._target_object = ""
        self._instruction = ""
        self._start_sim_time = 0.0
        self._episode_evidence = new_episode_evidence()
        self._last_error = ""

        self._prepare_dataset_root()
        callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            Image,
            str(self.get_parameter("wrist_rgb_topic").value),
            lambda message: self._image_callback("wrist", message),
            qos_profile_sensor_data,
            callback_group=callback_group,
        )
        if self.vla_wrist_rgb_topic:
            self.create_subscription(
                Image,
                self.vla_wrist_rgb_topic,
                lambda message: self._image_callback("vla_wrist", message),
                qos_profile_sensor_data,
                callback_group=callback_group,
            )
        self.create_subscription(
            Image,
            str(self.get_parameter("global_rgb_topic").value),
            lambda message: self._image_callback("global", message),
            qos_profile_sensor_data,
            callback_group=callback_group,
        )
        if self.global_right_rgb_topic:
            self.create_subscription(Image, self.global_right_rgb_topic,
                lambda message: self._image_callback("global_right", message),
                qos_profile_sensor_data, callback_group=callback_group)
        self.create_subscription(
            JointState,
            str(self.get_parameter("state_topic").value),
            self._state_callback,
            20,
            callback_group=callback_group,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("action_topic").value),
            self._action_callback,
            20,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("task_status_topic").value),
            self._task_status_callback,
            20,
            callback_group=callback_group,
        )
        status_qos = QoSProfile(depth=1)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String,
            "/smolvla/scene_reset_status",
            self._randomization_status_callback,
            status_qos,
            callback_group=callback_group,
        )
        self._status_pub = self.create_publisher(
            String, "/smolvla/recording_status", status_qos
        )
        self.create_service(
            Trigger,
            "/smolvla/recording/start",
            self._start_callback,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/recording/save",
            self._save_callback,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/recording/discard",
            self._discard_callback,
            callback_group=callback_group,
        )
        self._sample_timer = self.create_timer(
            1.0 / self.fps,
            self._sample_callback,
            callback_group=callback_group,
        )
        self._status_timer = self.create_timer(
            1.0, self._publish_status, callback_group=callback_group
        )
        self._publish_status()
        self.get_logger().info(
            f"Episode recorder ready: {self.dataset_root} at {self.fps:.1f} Hz"
        )

    def _prepare_dataset_root(self) -> None:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        info_path = self.dataset_root / "dataset_info.json"
        expected = {
            "format": FORMAT_VERSION,
            "fps": self.fps,
            "robot_type": "unitree_z1",
            "joint_names": list(JOINT_NAMES),
            "observation_keys": [
                "observation.images.wrist",
                *(
                    ["observation.images.vla_wrist"]
                    if "vla_wrist" in self.camera_names
                    else []
                ),
                "observation.images.global",
                *(["observation.images.global_right"] if "global_right" in self.camera_names else []),
                "observation.state",
            ],
            "action_key": "action",
            "image_encoding": "jpeg_bgr8",
            "image_shapes": {
                name: list(shape)
                for name, shape in self.expected_image_shapes.items()
            },
        }
        calibration_path = str(self.get_parameter("camera_calibration_path").value)
        if calibration_path and "global_right" in self.camera_names:
            expected["external_camera_calibration"] = json.loads(Path(calibration_path).read_text())
            expected["timestamp_domain"] = "simulation_clock"
        if info_path.exists():
            current = json.loads(info_path.read_text(encoding="utf-8"))
            keys = ["format", "fps", "joint_names", "image_shapes"]
            if "external_camera_calibration" in expected:
                keys.append("external_camera_calibration")
            for key in keys:
                if current.get(key) != expected[key]:
                    raise RuntimeError(
                        f"dataset metadata mismatch for {key}: "
                        f"{current.get(key)!r} != {expected[key]!r}"
                    )
            return
        self._write_json(info_path, expected)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _image_callback(self, name: str, message: Image) -> None:
        expected_shape = self.expected_image_shapes.get(name)
        actual_shape = (int(message.height), int(message.width), 3)
        if expected_shape is None or actual_shape != expected_shape:
            with self._lock:
                self._last_error = (
                    f"{name} image shape {actual_shape} does not match "
                    f"{expected_shape}"
                )
            return
        stamp = stamp_seconds(message.header.stamp)
        with self._lock:
            self._images[name] = (
                message,
                time.monotonic(),
                stamp,
            )
            # Synchronize lightweight ROS messages first.  Converting every
            # incoming 30--40 Hz frame to a NumPy/OpenCV image wastes work
            # because the recording grid only consumes 10 Hz.
            self._sync.push(name, stamp, message)

    def _update_joints(
        self, destination: Dict[str, float], message: JointState
    ) -> None:
        for name, value in zip(message.name, message.position):
            if name in JOINT_NAMES and np.isfinite(value):
                destination[name] = float(value)

    def _state_callback(self, message: JointState) -> None:
        with self._lock:
            self._update_joints(self._joint_state, message)
            if all(name in self._joint_state for name in JOINT_NAMES):
                self._joint_state_time = time.monotonic()
                self._joint_state_stamp = stamp_seconds(message.header.stamp)
            values = dict(zip(message.name, message.position))
            if all(name in values and np.isfinite(values[name]) for name in JOINT_NAMES):
                self._sync.push("state", stamp_seconds(message.header.stamp),
                                [float(values[name]) for name in JOINT_NAMES])

    def _action_callback(self, message: JointState) -> None:
        with self._lock:
            self._update_joints(self._action, message)
            if all(name in self._action for name in JOINT_NAMES):
                self._action_time = time.monotonic()
                self._action_stamp = stamp_seconds(message.header.stamp)
                self._sync.push("action", self._action_stamp,
                                [self._action[name] for name in JOINT_NAMES])

    def _task_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as exc:
            with self._lock:
                self._last_error = f"task status JSON: {exc}"
            return
        with self._lock:
            self._task_status = status
            self._task_status_time = time.monotonic()
            self._sync.push("status", self.get_clock().now().nanoseconds * 1.0e-9, dict(status))
            if self._recording:
                update_episode_evidence(self._episode_evidence, status)

    def _randomization_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as exc:
            with self._lock:
                self._last_error = f"scene randomization status JSON: {exc}"
            return
        with self._lock:
            self._randomization_status = status

    def _missing_streams(self, now: Optional[float] = None) -> Sequence[str]:
        current = time.monotonic() if now is None else now
        missing = []
        for name in self.camera_names:
            sample = self._images.get(name)
            if sample is None or current - sample[1] > self.stream_timeout:
                missing.append(f"{name}_rgb")
        if (
            not all(name in self._joint_state for name in JOINT_NAMES)
            or current - self._joint_state_time > self.stream_timeout
        ):
            missing.append("joint_states")
        if (
            not self._task_status
            or current - self._task_status_time > self.stream_timeout
        ):
            missing.append("task_status")
        return missing

    def _next_episode_index(self) -> int:
        indices = []
        for path in self.dataset_root.iterdir():
            name = path.name
            for prefix in ("episode_", ".recording_"):
                if name.startswith(prefix):
                    try:
                        indices.append(int(name[len(prefix):]))
                    except ValueError:
                        pass
        return max(indices, default=-1) + 1

    def _start_callback(self, _request, response):
        with self._lock:
            if self._recording:
                response.success = False
                response.message = "已经在录制 episode"
                return response
            missing = list(self._missing_streams())
            if missing:
                response.success = False
                response.message = "数据流未就绪: " + ", ".join(missing)
                self._last_error = response.message
                self._publish_status_locked()
                return response
            target = str(self._task_status.get("target_object", ""))
            if target not in TASK_INSTRUCTIONS:
                response.success = False
                response.message = f"无效训练目标: {target!r}"
                return response

            index = self._next_episode_index()
            episode_dir = self.dataset_root / f".recording_{index:06d}"
            episode_dir.mkdir(parents=False, exist_ok=False)
            for name in self.camera_names:
                (episode_dir / "images" / name).mkdir(parents=True)
            self._episode_randomization = {
                key: self._randomization_status.get(key)
                for key in (
                    "curriculum_level",
                    "target_grid",
                    "seed",
                    "object_positions_m",
                    "object_orientations_xyzw",
                    "object_displacements_mm",
                    "sweep_clearance_target",
                    "target_sweep_clear",
                )
                if key in self._randomization_status
            }
            self._write_json(
                episode_dir / "episode_start.json",
                {
                    "format": FORMAT_VERSION,
                    "episode_index": index,
                    "target_object": target,
                    "task": TASK_INSTRUCTIONS[target],
                    "fps": self.fps,
                    "joint_names": list(JOINT_NAMES),
                    "scene_randomization": self._episode_randomization,
                    "started_at_unix_s": time.time(),
                },
            )
            self._frames_file = (episode_dir / "frames.jsonl").open(
                "w", encoding="utf-8", buffering=1
            )
            self._recording = True
            self._episode_index = index
            self._episode_dir = episode_dir
            self._frame_count = 0
            self._dropped_frames = 0
            self._target_object = target
            self._instruction = TASK_INSTRUCTIONS[target]
            self._start_sim_time = self.get_clock().now().nanoseconds * 1.0e-9
            self._sync.begin_episode()
            self._sync_failure = ""
            self._selected_sample = None
            self._last_sync_clock = self._start_sim_time
            self._episode_evidence = new_episode_evidence()
            update_episode_evidence(
                self._episode_evidence, self._task_status
            )
            self._last_error = ""
            self._publish_status_locked()
            response.success = True
            response.message = f"开始录制 episode {index:06d}: {target}"
            return response

    def _sample_callback(self) -> None:
        with self._lock:
            now = self.get_clock().now().nanoseconds * 1.0e-9
            if self._last_sync_clock is not None and now < self._last_sync_clock - 1e-9:
                self._sync.reset()
                if self._recording:
                    self._sync_failure = "simulation clock moved backwards; discard and restart episode"
            self._last_sync_clock = now
            if not self._recording:
                return
            if self._sync_failure:
                self._last_error = self._sync_failure
                return
            target = self._start_sim_time + (self._frame_count + 1) / self.fps
            if now < target + self.sync_delay - 1e-9:
                return
            missing = list(self._missing_streams())
            current_target = str(
                self._task_status.get("target_object", "")
            )
            if current_target != self._target_object:
                missing.append("target_changed")
            if missing:
                self._dropped_frames += 1
                self._last_error = "跳过帧: " + ", ".join(missing)
                self._sync_failure = self._last_error
                return
            try:
                self._selected_sample = self._sync.select(target)
                if self._selected_sample["status"].get("target_object") != self._target_object:
                    raise ValueError("synchronized task target does not match episode")
                self._write_frame_locked()
            except (OSError, ValueError) as exc:
                self._dropped_frames += 1
                self._last_error = f"写入帧失败: {exc}"
                self._sync_failure = self._last_error
                self.get_logger().error(self._last_error)
            if self._frame_count % max(1, round(self.fps)) == 0:
                self._publish_status_locked()

    def _write_frame_locked(self) -> None:
        if self._episode_dir is None or self._frames_file is None:
            raise RuntimeError("recording paths are not initialized")
        sample = self._selected_sample
        if sample is None:
            raise ValueError("no synchronized sample selected")
        frame_index = self._frame_count
        filename = f"frame_{frame_index:06d}.jpg"
        image_paths = {
            name: self._episode_dir / "images" / name / filename
            for name in self.camera_names
        }
        options = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        converted_images = {}
        for name in image_paths:
            payload = sample["images"][name][1]
            image = (
                payload
                if isinstance(payload, np.ndarray)
                else image_to_bgr(payload)
            )
            expected_shape = getattr(self, "expected_image_shapes", {}).get(name)
            if expected_shape is not None and image.shape != expected_shape:
                raise ValueError(
                    f"{name} image shape {image.shape} does not match "
                    f"{expected_shape}"
                )
            converted_images[name] = image
        written_paths = []
        for name, image_path in image_paths.items():
            if not cv2.imwrite(str(image_path), converted_images[name], options):
                for written_path in written_paths:
                    written_path.unlink(missing_ok=True)
                raise OSError(f"failed to write {image_path}")
            written_paths.append(image_path)

        state = sample["state"]
        action = sample["action"][1]
        status = sample["status"]
        record = {
            "frame_index": frame_index,
            "timestamp": frame_index / self.fps,
            "simulation_time": sample["time"],
            "capture_time": self.get_clock().now().nanoseconds * 1.0e-9,
            "sync_tolerance_s": self.sync_tolerance,
            "joint_state_source_stamps": list(sample["state_bracket"]),
            "observation.images.wrist": str(
                image_paths["wrist"].relative_to(self._episode_dir)
            ),
            "observation.images.global": str(
                image_paths["global"].relative_to(self._episode_dir)
            ),
            "observation.state": state,
            "action": action,
            "joint_state_stamp": sample["time"],
            "action_stamp": sample["action"][0],
            "action_age_s": sample["time"] - sample["action"][0],
            "wrist_image_stamp": sample["images"]["wrist"][0],
            "global_image_stamp": sample["images"]["global"][0],
            "target_object": self._target_object,
            "task": self._instruction,
            "phase": str(status.get("phase", "")),
            "grasp_success": bool(status.get("grasp_success", False)),
            "physical_grasp_ok": bool(
                status.get("physical_grasp_ok", False)
            ),
            "object_touching_table": bool(
                status.get("object_touching_table", False)
            ),
            "task_success": bool(status.get("task_success", False)),
        }
        if "vla_wrist" in self.camera_names:
            record["observation.images.vla_wrist"] = str(
                image_paths["vla_wrist"].relative_to(self._episode_dir)
            )
            record["vla_wrist_image_stamp"] = sample["images"]["vla_wrist"][0]
        if "global_right" in self.camera_names:
            record["observation.images.global_right"] = str(image_paths["global_right"].relative_to(self._episode_dir))
            record["global_right_image_stamp"] = sample["images"]["global_right"][0]
        self._frames_file.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        self._sync.commit(sample)
        self._frame_count += 1
        self._last_error = ""

    def _save_callback(self, _request, response):
        with self._lock:
            if self._sync_failure:
                response.success = False
                response.message = "时间对齐失败，不能保存为连续训练回合，请丢弃后重录: " + self._sync_failure
                return response
            if not self._recording:
                response.success = False
                response.message = "当前没有正在录制的 episode"
                return response
            if self._frame_count < self.minimum_frames:
                response.success = False
                response.message = (
                    f"帧数不足: {self._frame_count} < {self.minimum_frames}"
                )
                return response
            update_episode_evidence(
                self._episode_evidence, self._task_status
            )
            failed_conditions = episode_save_failures(
                self._episode_evidence,
                self._task_status,
                self.require_task_success,
            )
            if failed_conditions:
                response.success = False
                response.message = (
                    "Episode 缺少完整纯物理抓取放置证据："
                    + ", ".join(failed_conditions)
                )
                return response
            index = int(self._episode_index)
            source = self._episode_dir
            if source is None:
                raise RuntimeError("recording path is unavailable")
            self._close_frames_file_locked()
            summary = {
                "synchronization": {
                    "method": "fixed_grid_nearest_rgb_interpolated_state_causal_action",
                    "tolerance_s": self.sync_tolerance,
                    "delay_s": self.sync_delay,
                    "action_age_s": self.action_age,
                },
                "format": FORMAT_VERSION,
                "episode_index": index,
                "target_object": self._target_object,
                "task": self._instruction,
                "fps": self.fps,
                "frame_count": self._frame_count,
                "duration_s": self._frame_count / self.fps,
                "dropped_frames": self._dropped_frames,
                "task_success": bool(
                    self._episode_evidence["task_success"]
                ),
                "physical_grasp_ok": bool(
                    self._episode_evidence[
                        "physical_grasp_verified"
                    ]
                ),
                "episode_goal": "physical_pick_place",
                "scene_randomization": self._episode_randomization,
                "bilateral_contact_ever": bool(
                    self._episode_evidence[
                        "bilateral_contact_ever"
                    ]
                ),
                "bilateral_contact_at_save": bool(
                    self._task_status.get("bilateral_contact", False)
                ),
                "object_lifted_clear_of_table_ever": bool(
                    self._episode_evidence["object_lifted_ever"]
                ),
                "object_lifted_at_save": bool(
                    self._task_status.get("lifted", False)
                ),
                "release_sequence_complete": bool(
                    self._episode_evidence[
                        "release_sequence_complete"
                    ]
                ),
                "placement_stable": bool(
                    self._episode_evidence["task_success"]
                ),
                "saved_at_unix_s": time.time(),
            }
            self._write_json(source / "episode_summary.json", summary)
            destination = self.dataset_root / f"episode_{index:06d}"
            os.replace(source, destination)
            with (self.dataset_root / "episodes.jsonl").open(
                "a", encoding="utf-8"
            ) as index_file:
                index_file.write(
                    json.dumps(summary, ensure_ascii=False) + "\n"
                )
                index_file.flush()
                os.fsync(index_file.fileno())
            frame_count = self._frame_count
            self._reset_recording_locked()
            self._publish_status_locked()
            response.success = True
            response.message = (
                f"已保存 episode {index:06d}，共 {frame_count} 帧"
            )
            return response

    def _discard_callback(self, _request, response):
        with self._lock:
            if not self._recording:
                response.success = False
                response.message = "当前没有正在录制的 episode"
                return response
            index = int(self._episode_index)
            episode_dir = self._episode_dir
            self._close_frames_file_locked()
            if episode_dir is not None and episode_dir.exists():
                shutil.rmtree(episode_dir)
            self._reset_recording_locked()
            self._publish_status_locked()
            response.success = True
            response.message = f"已丢弃 episode {index:06d}"
            return response

    def _close_frames_file_locked(self) -> None:
        if self._frames_file is not None:
            self._frames_file.flush()
            os.fsync(self._frames_file.fileno())
            self._frames_file.close()
            self._frames_file = None

    def _reset_recording_locked(self) -> None:
        self._recording = False
        self._sync_failure = ""
        self._selected_sample = None
        self._episode_index = None
        self._episode_dir = None
        self._frame_count = 0
        self._dropped_frames = 0
        self._target_object = ""
        self._instruction = ""
        self._start_sim_time = 0.0
        self._episode_evidence = new_episode_evidence()
        self._episode_randomization = {}
        self._last_error = ""

    def _status_payload_locked(self) -> Dict[str, Any]:
        missing = list(self._missing_streams())
        status = {
            "phase": "RECORDING" if self._recording else "IDLE",
            "ready": not missing,
            "recording": self._recording,
            "missing_streams": missing,
            "episode_index": self._episode_index,
            "frame_count": self._frame_count,
            "dropped_frames": self._dropped_frames,
            "target_object": self._target_object,
            "task": self._instruction,
            "curriculum_level": self._episode_randomization.get(
                "curriculum_level"
            ),
            "randomization_seed": self._episode_randomization.get("seed"),
            "task_success": bool(
                self._task_status.get("task_success", False)
            ),
            "dataset_root": str(self.dataset_root),
            "fps": self.fps,
            "sync_tolerance_s": self.sync_tolerance,
            "action_age_s": self.action_age,
            "last_error": self._last_error,
            "sync_failure": self._sync_failure,
        }
        if self._recording:
            status["duration_s"] = self._frame_count / self.fps
        return status

    def _publish_status_locked(self) -> None:
        message = String()
        message.data = json.dumps(
            self._status_payload_locked(), ensure_ascii=False
        )
        self._status_pub.publish(message)

    def _publish_status(self) -> None:
        with self._lock:
            self._publish_status_locked()

    def destroy_node(self) -> bool:
        with self._lock:
            if self._recording and self._episode_dir is not None:
                self._close_frames_file_locked()
                self._write_json(
                    self._episode_dir / "interrupted.json",
                    {
                        "episode_index": self._episode_index,
                        "frame_count": self._frame_count,
                        "interrupted_at_unix_s": time.time(),
                    },
                )
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeRecorder()
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


if __name__ == "__main__":
    main()
