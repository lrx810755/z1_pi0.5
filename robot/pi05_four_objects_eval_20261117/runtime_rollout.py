"""Run a trained SmolVLA checkpoint against the live Z1 Isaac Sim scene."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String
from rcl_interfaces.srv import GetParameters
from std_srvs.srv import SetBool, Trigger
from tf2_msgs.msg import TFMessage

from .dataset_selection import grid_bin
from .episode_recorder import image_to_bgr, stamp_seconds
from .recording_sync import RecordingSynchronizer
from .grasp_config import GRASP_CONFIG
from .smolvla_rollout_core import (
    BottleTipGuard,
    GraspPhaseTracker,
    PreliftContactGate,
    JOINT_NAMES,
    TASK_INSTRUCTIONS,
    arm_tracking_error,
    binary_gripper_target,
    GripperCloseHold,
    build_report,
    camera_keys_for_features,
    classify_failure,
    execution_window,
    load_training_seeds,
    pregrasp_collision_reason,
    safe_joint_target,
    streams_are_fresh,
    target_in_xy_bounds,
)
from .smolvla_state import AUGMENTED_STATE_NAMES, augmented_state


CAMERA_TOPICS = {
    "wrist": "/camera/camera/color/image_raw",
    "vla_wrist": "/vla_wrist_camera/color/image_raw",
    "global": "/global_camera/color/image_raw",
    "global_right": "/global_right_camera/color/image_raw",
}
DEFAULT_MANIFEST = Path(
    "/home/lp531/smolvla_data/z1_tabletop_reviewed_4x100_v1/selection_manifest.json"
)
DEFAULT_TRAINING_ROOT = Path("/home/lp531/smolvla_training")
GRID_CONFIG = GRASP_CONFIG["scene_randomization"]
GRID_WORKSPACE = (
    tuple(GRID_CONFIG["workspace_x_m"]),
    tuple(GRID_CONFIG["workspace_y_m"]),
)
GRID_SHAPE = tuple(GRID_CONFIG["levels"]["full"]["target_grid_shape"])


def load_training_coverage(root: Path):
    cells = {target: set() for target in TASK_INSTRUCTIONS}
    positions = {target: [] for target in TASK_INSTRUCTIONS}
    seeds = set()
    for episode in root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"):
        summary = json.loads((episode / "episode_summary.json").read_text(encoding="utf-8"))
        target = summary.get("target_object")
        if target not in cells or summary.get("task_success") is not True:
            continue
        randomization = summary["scene_randomization"]
        position = randomization["object_positions_m"][target]
        cells[target].add(grid_bin(position, GRID_WORKSPACE, GRID_SHAPE))
        positions[target].append((float(position[0]), float(position[1])))
        if randomization.get("seed") is not None:
            seeds.add(int(randomization["seed"]))
    if not any(cells.values()):
        raise ValueError(f"no successful training episodes in {root}")
    return cells, positions, seeds


def latest_checkpoint(root: Path = DEFAULT_TRAINING_ROOT) -> Path:
    candidates = list(root.glob("*/checkpoints/last/pretrained_model"))
    candidates.extend(root.glob("*/checkpoints/*/pretrained_model"))
    valid = [path for path in candidates if (path / "model.safetensors").is_file()]
    if not valid:
        raise FileNotFoundError(
            f"no completed SmolVLA checkpoint found below {root}; pass --checkpoint"
        )
    return max(valid, key=lambda path: path.stat().st_mtime)


class VideoSet:
    def __init__(self, directory: Path, fps: float) -> None:
        self.directory = directory
        self.fps = fps
        self.writers: dict[str, cv2.VideoWriter] = {}

    def write(self, images: dict[str, np.ndarray]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, image in images.items():
            if name not in self.writers:
                height, width = image.shape[:2]
                label = "global_third_person" if name == "global" else name
                path = self.directory / f"{label}.mp4"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not open video writer {path}")
                self.writers[name] = writer
            self.writers[name].write(image)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()


class SmolVLASimRollout(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("smolvla_sim_rollout")
        self.args = args
        self._lock = threading.RLock()
        self._images: dict[str, np.ndarray] = {}
        self._joints: dict[str, float] = {}
        self._stamps: dict[str, float] = {}
        self._status: dict[str, Any] = {}
        self._evidence: dict[str, bool] = {}
        self._randomization_status: dict[str, Any] = {}
        self._scene_grasp_reply: dict[str, Any] = {}
        self._seen_eval_seeds: set[int] = set()
        self._joint_sim_stamp = 0.0
        self._bottle_tip_guard = None
        self._box_geometry = None
        self.policy = self._load_policy(args.checkpoint)
        self._observation_sync = RecordingSynchronizer(
            self.camera_keys, tolerance=args.input_sync_tolerance, capacity=32,
        )
        self._input_sync_path = None
        self._input_sync_index = 0
        self._last_input_sync = None

        for name in self.camera_keys:
            topic = (
                args.wrist_rgb_topic if name == "wrist" else CAMERA_TOPICS[name]
            )
            self.create_subscription(
                Image, topic, lambda msg, camera=name: self._image_callback(camera, msg),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            JointState, "/joint_states", self._joint_callback, qos_profile_sensor_data
        )
        self.create_subscription(String, "/smolvla/status", self._status_callback, 10)
        self.create_subscription(
            TFMessage, "/isaac/object_transforms", self._bottle_pose_callback, 20
        )
        # Read the evaluator's actual box bounds, including launch overrides.
        self._box_parameter_client = self.create_client(
            GetParameters, "/smolvla_grasp_evaluator/get_parameters"
        )
        self._box_parameter_timer = self.create_timer(0.5, self._request_box_parameters)
        randomization_qos = QoSProfile(depth=10)
        randomization_qos.reliability = ReliabilityPolicy.RELIABLE
        randomization_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String, "/smolvla/scene_reset_status", self._randomization_callback,
            randomization_qos,
        )
        self._command_pub = self.create_publisher(JointState, "/isaac_joint_commands", 10)
        self._scene_grasp_pub = self.create_publisher(
            String, "/smolvla/scene_grasp_command", 10
        )
        self.create_subscription(
            String, "/smolvla/scene_grasp_status", self._scene_grasp_callback, 10
        )
        self._randomization_command_pub = self.create_publisher(
            String, "/smolvla/scene_randomization_command", 10
        )
        self._target_pub = self.create_publisher(String, "/smolvla/target_object", 10)
        active_qos = QoSProfile(depth=1)
        active_qos.reliability = ReliabilityPolicy.RELIABLE
        active_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._active_pub = self.create_publisher(
            Bool, "/smolvla/policy_control_active", active_qos
        )
        self._grasp_control_pub = self.create_publisher(
            String, "/smolvla/grasp_control_state", active_qos
        )
        self._randomize = self.create_client(
            Trigger, "/smolvla/randomize_episode_full"
        )
        self._reset_episode = self.create_client(Trigger, "/smolvla/reset_episode")
        self._stop_motion = self.create_client(Trigger, "/smolvla/stop_motion")
        self._servo_enable = self.create_client(
            SetBool, "/smolvla/enable_servo_stream"
        )

        if args.training_dataset_root is not None:
            self.covered_cells, self.training_target_positions, self.training_seeds = load_training_coverage(
                args.training_dataset_root
            )
        else:
            self.covered_cells = {}
            self.training_target_positions = {}
            self.training_seeds = (
                load_training_seeds(args.training_manifest)
                if args.training_manifest and args.training_manifest.is_file()
                else set()
            )

    def _load_policy(self, checkpoint: Path):
        import torch
        process_policy = self.args.policy_type in {"pi05", "diffusion"} or (
            self.args.policy_type == "act" and (checkpoint / "policy_preprocessor.json").is_file()
        )
        if self.args.policy_type == "diffusion":
            from .diffusion_policy_bridge import DiffusionProcessPolicy
            policy = DiffusionProcessPolicy(checkpoint)
        elif self.args.policy_type == "pi05":
            from .pi05_policy_bridge import PI05ProcessPolicy
            policy = PI05ProcessPolicy(checkpoint)
        elif self.args.policy_type == "act" and process_policy:
            from .act_policy_bridge import ACTProcessPolicy
            policy = ACTProcessPolicy(checkpoint)
        elif self.args.policy_type == "act":
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.act.modeling_act import ACTPolicy as Policy
            config = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
            config.pretrained_backbone_weights = None
        else:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
            config = None

        self.get_logger().info(f"Loading {self.args.policy_type} checkpoint: {checkpoint}")
        if not process_policy:
            policy = Policy.from_pretrained(
                str(checkpoint), config=config, local_files_only=True,
                strict=self.args.policy_type == "act",
            )
        policy.eval()
        configured_inputs = set(policy.config.input_features)
        if "observation.state" not in configured_inputs:
            raise ValueError(
                "checkpoint input schema does not match the recorded Z1 dataset: "
                "missing observation.state"
            )
        self.camera_keys = camera_keys_for_features(configured_inputs)
        action_feature = policy.config.output_features.get("action")
        if action_feature is None or tuple(action_feature.shape) != (len(JOINT_NAMES),):
            raise ValueError("checkpoint must produce seven absolute joint targets")
        state_feature = policy.config.input_features["observation.state"]
        self.state_dim = int(state_feature.shape[0])
        supported_state_dims = {len(JOINT_NAMES), len(AUGMENTED_STATE_NAMES)}
        if self.state_dim not in supported_state_dims:
            raise ValueError(
                "checkpoint observation.state must use the legacy 7-value or "
                f"augmented {len(AUGMENTED_STATE_NAMES)}-value schema, got {self.state_dim}"
            )
        self.device = torch.device(policy.config.device)
        self.use_amp = bool(policy.config.use_amp and self.device.type == "cuda")
        return policy

    def _image_callback(self, name: str, message: Image) -> None:
        try:
            image = image_to_bgr(message)
        except ValueError as exc:
            self.get_logger().error(f"Invalid {name} image: {exc}")
            return
        with self._lock:
            self._images[name] = image
            self._stamps[name] = time.monotonic()
            stamp = stamp_seconds(message.header.stamp)
            if self._joint_sim_stamp > 0 and stamp > self._joint_sim_stamp + 0.5:
                return  # Discard messages from a previous clock epoch.
            self._observation_sync.push(name, stamp, image)

    def _joint_callback(self, message: JointState) -> None:
        values = dict(zip(message.name, message.position))
        if not all(name in values for name in JOINT_NAMES):
            return
        positions = {name: float(values[name]) for name in JOINT_NAMES}
        if not all(np.isfinite(value) for value in positions.values()):
            return
        with self._lock:
            stamp = stamp_seconds(message.header.stamp)
            if stamp < self._joint_sim_stamp:
                # A delayed message cannot distinguish clock reset from DDS
                # reordering. Never roll safety feedback back; a simulator
                # clock reset requires restarting this validation process.
                return
            self._joints = positions
            self._stamps["joint"] = time.monotonic()
            self._joint_sim_stamp = stamp
            self._observation_sync.push("state", stamp, [positions[name] for name in JOINT_NAMES])

    def _request_box_parameters(self) -> None:
        if not self._box_parameter_client.service_is_ready():
            return
        self._box_parameter_timer.cancel()
        request = GetParameters.Request()
        request.names = [
            "box_center_x_m", "box_center_y_m", "box_inner_half_extent_m",
        ]
        self._box_parameter_client.call_async(request).add_done_callback(
            self._box_parameters_ready
        )

    def _box_parameters_ready(self, future) -> None:
        try:
            values = future.result().values
            geometry = tuple(float(value.double_value) for value in values)
            if len(geometry) != 3 or not all(np.isfinite(geometry)) or geometry[2] <= 0:
                raise ValueError("invalid evaluator box bounds")
            with self._lock:
                self._box_geometry = geometry
        except Exception as exc:
            self.get_logger().error(f"Cannot read placement bounds: {exc}")

    def _bottle_pose_callback(self, message: TFMessage) -> None:
        with self._lock:
            guard = self._bottle_tip_guard
            if guard is None:
                return
            for transform in message.transforms:
                frame = ''.join(c for c in transform.child_frame_id.lower() if c.isalnum())
                if not frame.endswith("beveragebottle"):
                    continue
                position = transform.transform.translation
                status = self._status
                verified = bool(status.get("was_grasped") or status.get("physical_grasp_ok")
                                or self._evidence.get("physical_grasp_ok"))
                over_box = bool(status.get("inside_success_volume"))
                if self._box_geometry is not None:
                    x, y, half_extent = self._box_geometry
                    half_extent = max(0.055, half_extent)
                    over_box = (abs(position.x - x) <= half_extent
                                and abs(position.y - y) <= half_extent)
                placing = (status.get("phase") in {"PLACING", "SUCCESS"}
                           or (verified and over_box))
                rotation = transform.transform.rotation
                guard.observe((rotation.x, rotation.y, rotation.z, rotation.w),
                              stamp_seconds(transform.header.stamp), placing)

    def _status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(status, dict):
            return
        with self._lock:
            self._status = status
            self._stamps["status"] = time.monotonic()
            gate = getattr(self, "_contact_gate", None)
            if gate is not None:
                gate.observe(status, self._joint_sim_stamp)
            for key in (
                "physical_grasp_ok", "lifted", "bilateral_contact",
                "release_sequence_complete", "task_success",
            ):
                self._evidence[key] = self._evidence.get(key, False) or bool(
                    status.get(key, False)
                )

    def _randomization_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(status, dict):
            with self._lock:
                self._randomization_status = status

    def _scene_grasp_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(status, dict) and status.get("phase") in (
            "FROZEN", "PHYSICS_READY", "ERROR"
        ):
            with self._lock:
                self._scene_grasp_reply = status

    def _set_target_frozen(self, target: str, frozen: bool) -> None:
        expected = "FROZEN" if frozen else "PHYSICS_READY"
        with self._lock:
            self._scene_grasp_reply = {}
        message = String()
        command = {
            "command": "freeze" if frozen else "unfreeze", "target": target
        }
        if frozen:
            command['keep_collisions'] = True
        message.data = json.dumps(command)
        self._scene_grasp_pub.publish(message)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                reply = dict(self._scene_grasp_reply)
            if reply.get("phase") == expected:
                if frozen and reply.get('target_collisions_enabled') is not True:
                    raise RuntimeError('scene_grasp: frozen target collision mode not verified')
                return
            if reply.get("phase") == "ERROR":
                raise RuntimeError(f"scene_grasp: {reply.get('message', 'error')}")
            time.sleep(0.01)
        raise RuntimeError(f"scene_grasp_timeout:{expected}")

    def _call_trigger(self, client, timeout_s: float) -> str:
        if not client.wait_for_service(timeout_sec=self.args.startup_timeout):
            raise RuntimeError(f"service unavailable: {client.srv_name}")
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise TimeoutError(f"service timed out: {client.srv_name}")
        response = future.result()
        if response is None or not response.success:
            detail = response.message if response else "no response"
            raise RuntimeError(f"{client.srv_name}: {detail}")
        return response.message

    def _disable_other_control(self) -> None:
        if self._servo_enable.wait_for_service(timeout_sec=2.0):
            request = SetBool.Request()
            request.data = False
            future = self._servo_enable.call_async(request)
            deadline = time.monotonic() + 3.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
        self._call_trigger(self._stop_motion, 5.0)

    def _snapshot(self):
        with self._lock:
            now = time.monotonic()
            if not streams_are_fresh(
                self._stamps, now, self.args.stale_timeout, self.camera_keys
            ):
                raise RuntimeError("stale_stream")
            if (self.args.prelift_contact_stable_s > 0 and
                    now - self._stamps.get("status", 0.0) > self.args.stale_timeout):
                raise RuntimeError("stale_grasp_status")
            images = {name: self._images[name].copy() for name in self.camera_keys}
            joints = np.asarray([self._joints[name] for name in JOINT_NAMES])
            status = dict(self._status)
            evidence = dict(self._evidence)
        if self.args.reject_blank_cameras:
            for name, image in images.items():
                if image[::16, ::16].std() < 5.0:
                    raise RuntimeError(f"blank_camera:{name}")
        return images, joints, status, evidence

    def _synchronized_policy_observation(self):
        started = time.monotonic()
        deadline = started + self.args.input_sync_timeout
        reason = "missing streams"
        while time.monotonic() < deadline:
            with self._lock:
                if streams_are_fresh(self._stamps, time.monotonic(),
                                     self.args.stale_timeout, self.camera_keys):
                    try:
                        sample = self._observation_sync.select_latest_observation(
                            self._joint_sim_stamp, self.args.input_sync_max_age,
                        )
                        if self.args.policy_type == "diffusion":
                            from .diffusion_history import previous_observation
                            previous = previous_observation(self._observation_sync, sample)
                            self._diffusion_history_sample = previous
                    except ValueError as exc:
                        reason = str(exc)
                    else:
                        self._observation_sync.commit(sample)
                        stamps = {name: value[0] for name, value in sample["images"].items()}
                        all_stamps = [sample["time"], *stamps.values()]
                        self._last_input_sync = {
                            "prediction": self._input_sync_index,
                            "observation_time_s": sample["time"],
                            "image_stamps_s": stamps,
                            "state_bracket_s": list(sample["state_bracket"]),
                            "sensor_skew_s": max(all_stamps) - min(all_stamps),
                            "oldest_sensor_age_s": self._joint_sim_stamp - min(all_stamps),
                            "latest_state_time_s": self._joint_sim_stamp,
                            "latest_received_sensor_skew_s": max(
                                self._observation_sync.buffers[name][-1][0]
                                for name in (*self.camera_keys, "state")
                            ) - min(
                                self._observation_sync.buffers[name][-1][0]
                                for name in (*self.camera_keys, "state")
                            ),
                            "wait_wall_s": time.monotonic() - started,
                            "input_state": sample["state"],
                        }
                        if self.args.policy_type == "diffusion":
                            self._last_input_sync["history_observation_time_s"] = previous["time"]
                            self._last_input_sync["history_image_stamps_s"] = {k:v[0] for k,v in previous["images"].items()}
                            self._last_input_sync["history_state"] = previous["state"]
                        self._input_sync_index += 1
                        break
                else:
                    reason = "stale_stream"
            time.sleep(0.005)
        else:
            if self._input_sync_path is not None:
                with self._input_sync_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"error": "input_sync_timeout", "detail": reason}) + "\n")
            raise RuntimeError("input_sync_timeout: " + reason)
        if self._input_sync_path is not None:
            with self._input_sync_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self._last_input_sync) + "\n")
        images = {name: value[1] for name, value in sample["images"].items()}
        if self.args.reject_blank_cameras:
            for name, image in images.items():
                if image[::16, ::16].std() < 5.0:
                    raise RuntimeError(f"blank_camera:{name}")
        return images, np.asarray(sample["state"], dtype=np.float32)

    def _policy_batch(
        self,
        images: dict[str, np.ndarray],
        joints,
        status: dict[str, Any],
        task: str,
    ):
        import torch

        images, joints = self._synchronized_policy_observation()
        state = (
            np.asarray(augmented_state(joints, status), dtype=np.float32)
            if self.state_dim == len(AUGMENTED_STATE_NAMES)
            else joints
        )
        batch: dict[str, Any] = {
            "observation.state": torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0),
        }
        if self.args.policy_type in {"smolvla", "pi05"}:
            batch["task"] = [task]
        for name in self.camera_keys:
            rgb = cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(
                device=self.device, dtype=torch.float32
            )
            batch[f"observation.images.{name}"] = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        if self.args.policy_type == "diffusion":
            previous = self._diffusion_history_sample
            batch["observation.state"] = torch.stack([
                torch.as_tensor(previous["state"], dtype=torch.float32, device=self.device),
                batch["observation.state"][0],
            ], dim=0).unsqueeze(0)
            for name in self.camera_keys:
                image = previous["images"][name][1]
                if self.args.reject_blank_cameras and image[::16, ::16].std() < 5.0:
                    raise RuntimeError(f"blank_history_camera:{name}")
                rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                old = torch.from_numpy(rgb).to(device=self.device, dtype=torch.float32).permute(2,0,1)/255.0
                key = f"observation.images.{name}"
                batch[key] = torch.stack([old, batch[key][0]], dim=0).unsqueeze(0)
        return batch

    def _predict(self, batch) -> np.ndarray:
        import torch

        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp else nullcontext()
        )
        with torch.inference_mode(), context:
            chunk = self.policy.predict_action_chunk(batch)
        full_chunk = chunk.detach().float().cpu().numpy()
        trace_path = getattr(self, "_prediction_trace_path", None)
        if trace_path is not None:
            gripper = full_chunk[0, :, -1] if full_chunk.ndim == 3 else full_chunk[:, -1]
            first_close = next((i for i, value in enumerate(gripper) if value >= 0.6), None)
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "prediction": getattr(self, "_prediction_trace_index", 0),
                    "sim_stamp_s": self._joint_sim_stamp,
                    "execute_steps": self.args.execute_steps,
                    "input_state": batch["observation.state"].detach().cpu().numpy().reshape(-1).tolist(),
                    "gripper_prediction": gripper.tolist(),
                    "first_close_step_06": first_close,
                    "close_in_execution_window": first_close is not None and first_close < self.args.execute_steps,
                }) + "\n")
            self._prediction_trace_index = getattr(self, "_prediction_trace_index", 0) + 1
        return execution_window(full_chunk, self.args.execute_steps)

    def _publish_target(self, target: str) -> None:
        message = String()
        message.data = target
        for _ in range(3):
            self._target_pub.publish(message)
            time.sleep(0.05)

    def _publish_command(self, positions) -> None:
        message = JointState()
        # This standalone node may use wall time. Recorded commands must share
        # the Isaac sensor clock or they poison the causal-action buffer.
        with self._lock:
            command_ns = round(self._joint_sim_stamp * 1_000_000_000)
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(
            command_ns, 1_000_000_000
        )
        message.name = list(JOINT_NAMES)
        message.position = [float(value) for value in positions]
        self._command_pub.publish(message)

    def _set_active(self, value: bool) -> None:
        message = Bool()
        message.data = value
        self._active_pub.publish(message)

    def _publish_grasp_phase(
        self, target: str, phase: dict[str, object]
    ) -> None:
        message = String()
        message.data = json.dumps(
            {"target_object": target, **phase},
            ensure_ascii=False,
            sort_keys=True,
        )
        self._grasp_control_pub.publish(message)

    def _wait_sim_period(self, start_stamp: float) -> None:
        target_stamp = start_stamp + 1.0 / (self.args.execution_rate or self.args.rate)
        wall_deadline = time.monotonic() + self.args.sim_clock_timeout
        while time.monotonic() < wall_deadline:
            with self._lock:
                current_stamp = self._joint_sim_stamp
            if current_stamp >= target_stamp:
                return
            time.sleep(0.005)
        raise RuntimeError("simulation_clock_stalled")

    def _target_in_covered_cell(self, status: dict, target: str) -> bool:
        if not self.covered_cells:
            return True
        try:
            position = status["object_positions_m"][target]
            cell = grid_bin(position, GRID_WORKSPACE, GRID_SHAPE)
        except (KeyError, IndexError, TypeError, ValueError):
            return False
        if cell not in self.covered_cells[target]:
            return False
        radius = self.args.max_training_target_distance_m
        if radius is None:
            return True
        x, y = position[:2]
        return any(
            np.hypot(x - train_x, y - train_y) <= radius
            for train_x, train_y in self.training_target_positions[target]
        )

    def _randomize_scene(self, target: str, fixed_seed: int | None = None) -> int:
        self._set_active(False)
        if fixed_seed is not None:
            if fixed_seed in self.training_seeds:
                raise ValueError(f"evaluation seed {fixed_seed} was used for training")
            self._call_trigger(self._reset_episode, 120.0)
            with self._lock:
                self._randomization_status = {}
                self._evidence = {}
                self._status = {}
            message = String()
            message.data = json.dumps({
                "seed": fixed_seed, "level": "full", "target_grid": None,
                "settle_duration_s": 2.0,
                "non_target_max_angular_speed_rad_s": 0.35,
            })
            self._randomization_command_pub.publish(message)
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                with self._lock:
                    status = dict(self._randomization_status)
                if status.get("seed") == fixed_seed and status.get("phase") == "RANDOMIZED":
                    if not self._target_in_covered_cell(status, target):
                        raise ValueError(f"evaluation seed {fixed_seed} is outside training cells")
                    if self.args.target_xy_bounds and not target_in_xy_bounds(
                        status, target, self.args.target_xy_bounds
                    ):
                        raise ValueError(f"evaluation seed {fixed_seed} is outside training XY bounds")
                    return fixed_seed
                if status.get("phase") == "ERROR":
                    raise RuntimeError(f"scene randomization failed: {status}")
                time.sleep(0.02)
            raise TimeoutError(f"scene randomization timed out for seed {fixed_seed}")
        for _attempt in range(50 if self.covered_cells else 10):
            with self._lock:
                self._randomization_status = {}
                self._evidence = {}
                self._status = {}
            try:
                self._call_trigger(self._randomize, 120.0)
            except (RuntimeError, TimeoutError) as exc:
                self.get_logger().warning(
                    f"Full randomization rejected; resampling: {exc}"
                )
                continue
            status_deadline = time.monotonic() + 3.0
            randomization = {}
            while time.monotonic() < status_deadline:
                with self._lock:
                    randomization = dict(self._randomization_status)
                if (
                    randomization.get("phase") == "RANDOMIZED"
                    and randomization.get("seed") is not None
                ):
                    break
                time.sleep(0.02)
            seed = randomization.get("seed")
            if seed is None:
                raise RuntimeError("randomization completed without a reported seed")
            seed = int(seed)
            if not self._target_in_covered_cell(randomization, target):
                self.get_logger().info(f"Resampling {target} outside training cells")
                continue
            if self.args.target_xy_bounds and not target_in_xy_bounds(
                randomization, target, self.args.target_xy_bounds
            ):
                self.get_logger().info(
                    f"Resampling bottle outside training XY bounds: "
                    f"{randomization.get('object_positions_m', {}).get('beverage_bottle')}"
                )
                continue
            if seed not in self.training_seeds and seed not in self._seen_eval_seeds:
                self._seen_eval_seeds.add(seed)
                return seed
            self.get_logger().warning(f"Resampling previously used seed {seed}")
        raise RuntimeError("could not obtain an unseen randomization seed")

    def _activate_policy(self, target: str) -> None:
        self._disable_other_control()
        with self._lock:
            self._evidence = {}
            self._status = {}
            self._observation_sync.reset()
            self._last_input_sync = None
        self._publish_target(target)
        self.policy.reset()
        stream_deadline = time.monotonic() + 5.0
        ready = False
        while time.monotonic() < stream_deadline:
            with self._lock:
                ready = streams_are_fresh(
                    self._stamps,
                    time.monotonic(),
                    self.args.stale_timeout,
                    self.camera_keys,
                )
            if ready:
                break
            time.sleep(0.02)
        if not ready:
            raise RuntimeError("camera or joint streams did not recover after reset")
        self._set_active(True)
        time.sleep(0.05)

    def _prepare_episode(self, target: str, fixed_seed: int | None = None) -> int:
        self._publish_target(target)
        seed = self._randomize_scene(target, fixed_seed)
        self._activate_policy(target)
        return seed

    def _episode_seed(
        self, target: str, prepared_seed: int | None,
        fixed_seed: int | None = None,
    ) -> int:
        if prepared_seed is not None and fixed_seed is not None:
            raise ValueError("prepared seed and fixed seed cannot both be set")
        if prepared_seed is not None:
            return int(prepared_seed)
        return (
            self._prepare_episode(target, fixed_seed)
            if fixed_seed is not None else self._prepare_episode(target)
        )

    def run_episode(
        self,
        target: str,
        episode_number: int,
        root: Path,
        prepared_seed: int | None = None,
        stop_event: threading.Event | None = None,
        fixed_seed: int | None = None,
    ) -> dict:
        close_hold = GripperCloseHold(getattr(self.args, "gripper_min_close_hold_s", 0.0))
        previous_gripper_target = -0.90  # Reset hysteresis for each episode.
        started = time.monotonic()
        seed = self._episode_seed(target, prepared_seed, fixed_seed)
        with self._lock:
            initial_scene_randomization = dict(self._randomization_status)
            initial_joint_state = [self._joints.get(name) for name in JOINT_NAMES]
            target_position = initial_scene_randomization.get(
                "object_positions_m", {}
            ).get(target)
        if prepared_seed is not None:
            self._activate_policy(target)
        episode_dir = root / "episodes" / f"{target}_{episode_number:03d}_seed_{seed}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        self._prediction_trace_path = episode_dir / "prediction_chunks.jsonl"
        self._prediction_trace_index = 0
        self._input_sync_path = episode_dir / "input_sync.jsonl"
        self._input_sync_index = 0
        video = VideoSet(episode_dir / "videos", self.args.rate)
        trace_path = episode_dir / "trace.jsonl"
        success = False
        stop_reason = ""
        final_status: dict[str, Any] = {}
        final_evidence: dict[str, bool] = {}
        first_pregrasp_contact: dict[str, Any] | None = None
        grasp_phase = GraspPhaseTracker()
        execution_rate = self.args.execution_rate or self.args.rate
        substeps = round(execution_rate / self.args.rate)
        contact_gate = PreliftContactGate(
            self.args.prelift_contact_stable_s, self.args.prelift_contact_timeout_s, self.args.stale_timeout
        )
        with self._lock:
            self._contact_gate = contact_gate
            self._bottle_tip_guard = (
                BottleTipGuard(self._joint_sim_stamp) if target == "beverage_bottle" else None
            )
        gripper_closed_once = False
        target_frozen = False
        self._set_active(True)
        try:
            if self.args.freeze_target_until_close:
                # Mark before sending so a lost acknowledgement still triggers cleanup.
                target_frozen = True
                self._set_target_frozen(target, True)
            with self._lock:
                episode_start_sim = self._joint_sim_stamp
            deadline_sim = episode_start_sim + self.args.timeout
            last_command = None
            tracking_error_since = None
            with trace_path.open("w", encoding="utf-8") as trace:
                def stop_on_pregrasp_collision(images, joints, status, requested=None):
                    nonlocal stop_reason, first_pregrasp_contact
                    if not self.args.stop_on_pregrasp_contact or gripper_closed_once:
                        return False
                    gripper_command = (
                        grasp_phase.previous_command
                        if grasp_phase.previous_command is not None
                        else float(joints[-1])
                    )
                    reason = pregrasp_collision_reason(
                        status, target, gripper_command, target_position
                    )
                    if reason is None:
                        return False
                    stop_reason = reason
                    first_pregrasp_contact = {
                        "reason": reason,
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "joint_state": np.asarray(joints).tolist(),
                        "last_command": (
                            last_command.tolist() if last_command is not None else None
                        ),
                        "next_policy_action": (
                            np.asarray(requested).tolist() if requested is not None else None
                        ),
                        "target_position_m": target_position,
                        "object_position_m": status.get("object_position_m"),
                        "gripper_object_distance_m": status.get("gripper_object_distance_m"),
                        "relative_position_gripper_m": status.get("relative_position_gripper_m"),
                        "left_finger_contact_object": status.get("left_finger_contact_object"),
                        "right_finger_contact_object": status.get("right_finger_contact_object"),
                        "left_finger_contact_impulse_ns": status.get("left_finger_contact_impulse_ns"),
                        "right_finger_contact_impulse_ns": status.get("right_finger_contact_impulse_ns"),
                    }
                    (episode_dir / "first_contact.json").write_text(
                        json.dumps(first_pregrasp_contact, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    video.write(images)
                    return True

                def stop_on_bottle_tip():
                    nonlocal stop_reason
                    with self._lock:
                        guard = self._bottle_tip_guard
                        tipped = guard is not None and guard.tripped
                    if tipped:
                        stop_reason = "bottle_tipped"
                        self.get_logger().error(
                            "瓶子倾斜大于70°并持续1秒，停止本轮评估：失败"
                        )
                    return tipped

                while True:
                    if stop_on_bottle_tip():
                        break
                    if stop_event is not None and stop_event.is_set():
                        stop_reason = "operator_stop"
                        break
                    with self._lock:
                        if self._joint_sim_stamp >= deadline_sim:
                            break
                    images, joints, status, evidence = self._snapshot()
                    final_status, final_evidence = status, evidence
                    if bool(status.get("task_success", False)):
                        success = True
                        break
                    if stop_on_pregrasp_collision(images, joints, status):
                        break
                    if bool(status.get("left_finger_touching_table", False)) or bool(
                        status.get("right_finger_touching_table", False)
                    ):
                        stop_reason = "finger_table_contact"
                        break
                    video.write(images)
                    actions = self._predict(
                        self._policy_batch(
                            images, joints, status, TASK_INSTRUCTIONS[target]
                        )
                    )
                    for action_index, requested in enumerate(actions):
                        execution_target = (
                            binary_gripper_target(requested, self.args.gripper_binary_threshold_rad,
                                                  self.args.gripper_action_encoding,
                                                  previous_target=previous_gripper_target)
                            if self.args.binary_gripper else requested
                        )
                        if self.args.binary_gripper:
                            execution_target[-1] = close_hold.apply(float(execution_target[-1]), self._joint_sim_stamp)
                            previous_gripper_target = float(execution_target[-1])
                        with self._lock:
                            interval_start = self._joint_sim_stamp
                        _, segment_current, _, _ = self._snapshot()
                        segment_start = last_command.copy() if last_command is not None else segment_current.copy()
                        endpoint = safe_joint_target(execution_target, segment_current, self.args.max_joint_step)
                        gripper_target = execution_target if self.args.binary_gripper else endpoint
                        substep = 0
                        gripper_advanced = False
                        while substep < substeps:
                            if stop_event is not None and stop_event.is_set():
                                stop_reason = "operator_stop"
                                break
                            with self._lock:
                                deadline_reached = (
                                    self._joint_sim_stamp >= deadline_sim
                                )
                            if deadline_reached:
                                break
                            if substeps > 1:
                                self._wait_sim_period(interval_start + substep / execution_rate)
                                if stop_event is not None and stop_event.is_set():
                                    stop_reason = "operator_stop"
                                    break
                            if stop_on_bottle_tip():
                                break
                            images, current, status, evidence = self._snapshot()
                            final_status, final_evidence = status, evidence
                            if stop_on_pregrasp_collision(images, current, status, execution_target):
                                break
                            if last_command is not None and arm_tracking_error(
                                last_command, current
                            ) > self.args.tracking_error:
                                tracking_error_since = tracking_error_since or time.monotonic()
                                if (
                                    time.monotonic() - tracking_error_since
                                    >= self.args.tracking_error_duration
                                ):
                                    stop_reason = "joint_tracking_fault"
                                    break
                            else:
                                tracking_error_since = None
                            if bool(status.get("task_success", False)):
                                success = True
                                break
                            if bool(status.get("left_finger_touching_table", False)) or bool(
                                status.get("right_finger_touching_table", False)
                            ):
                                stop_reason = "finger_table_contact"
                                break
                            command = safe_joint_target(
                                (segment_start + (endpoint - segment_start) * ((substep + 1) / substeps))
                                if substeps > 1 else execution_target,
                                current, self.args.max_joint_step
                            )
                            if self.args.binary_gripper:
                                # Keep approaching the binary goal on every command tick;
                                # arm interpolation and the per-command safety limit remain active.
                                command[-1] = safe_joint_target(
                                    execution_target, current, self.args.max_joint_step
                                )[-1]
                            if gripper_advanced:
                                command[-1] = safe_joint_target(gripper_target, current, self.args.max_joint_step)[-1]
                            with self._lock:
                                gate_stamp = self._joint_sim_stamp
                                hold_arm = contact_gate.hold_arm if contact_gate.hold_arm is not None else current[:6]
                                arm_segment_moves = bool(np.max(np.abs(endpoint[:6] - hold_arm)) > 0.002)
                                if (contact_gate.stable_s > 0 and not contact_gate.ready
                                        and arm_segment_moves and command[-1] >= -0.75):
                                    # Closing must continue while arm progress is paused.
                                    gripper_advanced = True
                                    command[-1] = safe_joint_target(gripper_target, current, self.args.max_joint_step)[-1]
                                command = contact_gate.apply(command, current, gate_stamp)
                                source_wait = (contact_gate.started is not None
                                               and not contact_gate.ready and arm_segment_moves)
                            if command[-1] >= -0.75:
                                gripper_closed_once = True
                            phase = grasp_phase.update(
                                command[-1], current[-1], status,
                                gripper_target=(execution_target[-1] if self.args.binary_gripper else None),
                            )
                            if target_frozen and phase["gripper_commanded_closed"]:
                                self._set_target_frozen(target, False)
                                target_frozen = False
                            self._publish_grasp_phase(target, phase)
                            with self._lock:
                                sim_stamp = self._joint_sim_stamp
                            self._publish_command(command)
                            last_command = command
                            if substep == substeps - 1 and not source_wait:
                                video.write(images)
                            trace.write(json.dumps({
                                "elapsed_s": time.monotonic() - started,
                                "simulation_elapsed_s": sim_stamp - episode_start_sim,
                                "joint_state": current.tolist(),
                                "policy_action": np.asarray(requested).tolist(),
                                "execution_target": np.asarray(execution_target).tolist(),
                                "binary_gripper": self.args.binary_gripper,
                                "gripper_close_hold_blocked_open": close_hold.blocked_open,
                                "command": command.tolist(),
                                "phase": status.get("phase"),
                                "policy_action_index": action_index,
                                "execution_substep": substep,
                                "execution_rate_hz": execution_rate,
                                "prelift_waiting": contact_gate.started is not None and not contact_gate.ready,
                                "policy_action_paused": source_wait,
                                "command_target_frozen": target_frozen,
                                "command_target_collisions_enabled": status.get("target_collisions_enabled"),
                                "physical_grasp_ok": bool(evidence.get("physical_grasp_ok")),
                                "lifted": bool(evidence.get("lifted")),
                                "release_sequence_complete": bool(
                                    evidence.get("release_sequence_complete")
                                ),
                                "contact_status": {key: status.get(key) for key in (
                                    "left_finger_contact_object", "right_finger_contact_object",
                                    "left_finger_touching_table", "right_finger_touching_table",
                                    "object_position_m", "gripper_object_distance_m",
                                    "relative_position_gripper_m", "frozen",
                                    "gripper_tf_stamp_s", "object_pose_stamp_s",
                                    "gripper_world_position_m", "gripper_world_orientation_xyzw",
                                    "target_collisions_enabled",
                                    "target_finger_min_separation_m",
                                    "target_finger_max_penetration_m",
                                    "target_finger_contact_pair_count",
                                    "object_orientation_xyzw", "object_speed_m_s",
                                    "cylinder_tilt_deg", "cylinder_tilt_valid",
                                    "contact_angle_valid", "geometry_contact_valid",
                                    "left_finger_contact_impulse_ns", "right_finger_contact_impulse_ns",
                                    "contact_report_available", "gripper_obstructed",
                                    "gripper_commanded_closed", "inside_grasp_envelope",
                                    "grasp_envelope", "relative_pose_stable",
                                    "object_touching_table", "physical_test_lift_active",
                                    "physical_grasp_candidate", "physical_grasp_ok", "was_grasped",
                                    "release_started", "release_sequence_complete",
                                    "inside_success_volume", "failure_reason",
                                )},
                            }, ensure_ascii=False) + "\n")
                            trace.flush()
                            if substeps == 1:
                                self._wait_sim_period(sim_stamp)
                            if source_wait:
                                interval_start += 1.0 / execution_rate
                            else:
                                substep += 1
                        if success or stop_reason:
                            break
                    if success or stop_reason:
                        break
        except (RuntimeError, ValueError) as exc:
            error_detail = str(exc) if str(exc) else type(exc).__name__
            if error_detail == "stale_stream":
                stop_reason = "stale_stream"
            elif error_detail.startswith("input_sync_timeout:"):
                stop_reason = "input_sync_timeout"
            elif error_detail.startswith("blank_camera:"):
                stop_reason = "blank_camera"
            elif isinstance(exc, ValueError):
                stop_reason = "invalid_policy_action"
            elif error_detail in {"prelift_contact_not_stable", "stale_grasp_status"}:
                stop_reason = error_detail
            else:
                stop_reason = "runtime_error"
        finally:
            # Stop before scene cleanup, which can block waiting for an ACK.
            with self._lock:
                hold = [self._joints.get(name) for name in JOINT_NAMES]
            if all(value is not None and np.isfinite(value) for value in hold):
                self._publish_command(hold)
            self._set_active(False)
            with self._lock:
                self._bottle_tip_guard = None
            if target_frozen:
                try:
                    self._set_target_frozen(target, False)
                except RuntimeError as exc:
                    self.get_logger().error(f"Target unfreeze failed: {exc}")
                    cleanup_error = str(exc)
                    if not stop_reason:
                        stop_reason = "target_unfreeze_failed"
                        error_detail = cleanup_error
                    success = False
            self._publish_grasp_phase(target, {
                "gripper_commanded_closed": False,
                "gripper_obstructed": False,
                "physical_test_lift_active": False,
                "physical_grasp_ok": False,
                "release_started": False,
                "release_sequence_complete": False,
                "transport_failure_reason": stop_reason,
            })
            video.close()

        timed_out = not success and not stop_reason
        merged_status = {**final_status, **final_evidence, "task_success": success}
        category = classify_failure(merged_status, timed_out, stop_reason)
        result = {
            "target_object": target,
            "episode_number": episode_number,
            "seed": seed,
            "target_position_m": target_position,
            "initial_joint_state": initial_joint_state,
            "initial_scene_randomization": initial_scene_randomization,
            "task": TASK_INSTRUCTIONS[target],
            "success": success,
            "failure_category": category,
            "stop_reason": stop_reason,
            "error_detail": locals().get("error_detail", ""),
            "cleanup_error": locals().get("cleanup_error", ""),
            "duration_s": round(time.monotonic() - started, 3),
            "simulation_duration_s": round(
                max(0.0, self._joint_sim_stamp - locals().get(
                    "episode_start_sim", self._joint_sim_stamp
                )),
                3,
            ),
            "physical_predicates": final_evidence,
            "first_pregrasp_contact": first_pregrasp_contact,
            "action_rate_hz": self.args.rate,
            "execution_rate_hz": execution_rate,
            "execute_steps": self.args.execute_steps,
            "binary_gripper": self.args.binary_gripper,
            "gripper_action_encoding": self.args.gripper_action_encoding,
            "gripper_min_close_hold_s": getattr(self.args, "gripper_min_close_hold_s", 0.0),
            "gripper_binary_threshold": None if self.args.gripper_action_encoding == "binary" else self.args.gripper_binary_threshold_rad,
            "gripper_binary_hysteresis": ({"open_threshold": 0.4, "close_threshold": 0.6,
                                            "initial_target_rad": -0.9}
                                           if self.args.binary_gripper and self.args.gripper_action_encoding == "binary" else None),
            "gripper_binary_threshold_rad": self.args.gripper_binary_threshold_rad,
            "input_sync_tolerance_s": self.args.input_sync_tolerance,
            "input_sync_max_age_s": self.args.input_sync_max_age,
            "prelift_contact_stable_s": self.args.prelift_contact_stable_s,
            "freeze_target_until_close": self.args.freeze_target_until_close,
            "frozen_target_collision_mode": "enabled" if self.args.freeze_target_until_close else "not_frozen",
            "final_status": final_status,
            "visual_review_required": episode_number <= self.args.preview_episodes,
        }
        (episode_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    def run_suite(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = self.args.output / f"{self.args.policy_type}_sim_eval_{stamp}"
        root.mkdir(parents=True, exist_ok=False)
        results = []
        try:
            for target in self.args.objects:
                for number in range(1, self.args.episodes_per_object + 1):
                    self.get_logger().info(
                        f"Starting {target} episode {number}/{self.args.episodes_per_object}"
                    )
                    try:
                        fixed_seed = (
                            self.args.eval_seeds[number - 1]
                            if self.args.eval_seeds else None
                        )
                        result = self.run_episode(
                            target, number, root, fixed_seed=fixed_seed
                        )
                    except Exception as exc:
                        self._set_active(False)
                        result = {
                            "target_object": target,
                            "episode_number": number,
                            "seed": None,
                            "task": TASK_INSTRUCTIONS[target],
                            "success": False,
                            "failure_category": "episode_setup_failed",
                            "stop_reason": "episode_setup_failed",
                            "error_detail": str(exc),
                            "duration_s": 0.0,
                            "physical_predicates": {},
                            "final_status": {},
                            "visual_review_required": (
                                number <= self.args.preview_episodes
                            ),
                        }
                        failure_dir = root / "episodes" / (
                            f"{target}_{number:03d}_setup_failed"
                        )
                        failure_dir.mkdir(parents=True, exist_ok=True)
                        (failure_dir / "result.json").write_text(
                            json.dumps(result, ensure_ascii=False, indent=2)
                            + "\n",
                            encoding="utf-8",
                        )
                    results.append(result)
                    report = {
                        "checkpoint": str(self.args.checkpoint),
                        "policy_type": self.args.policy_type,
                        "target_xy_bounds": self.args.target_xy_bounds,
                        **build_report(
                            results,
                            expected_objects=self.args.objects,
                            expected_per_object=self.args.episodes_per_object,
                        ),
                    }
                    (root / "report.json").write_text(
                        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    self.get_logger().info(
                        f"Finished {target} episode {number}: {result['failure_category']}"
                    )
        finally:
            self._set_active(False)
        return root


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy in the running Z1 Isaac Sim scene."
    )
    parser.add_argument("--policy-type", choices=("smolvla", "act", "pi05", "diffusion"), default="smolvla")
    parser.add_argument(
        "--wrist-rgb-topic", default=CAMERA_TOPICS["wrist"],
        help="ROS image topic for the dataset's observation.images.wrist view",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--target-xy-bounds", nargs=4, type=float,
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument(
        "--objects", nargs="+", choices=tuple(TASK_INSTRUCTIONS),
        default=list(TASK_INSTRUCTIONS),
    )
    parser.add_argument("--episodes-per-object", type=int, default=20)
    parser.add_argument("--eval-seeds", nargs="+", type=int)
    parser.add_argument("--gripper-min-close-hold-s", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--execution-rate", type=float,
                        help="Command frequency; integer multiple of --rate; defaults to --rate")
    parser.add_argument("--prelift-contact-stable-s", type=float, default=0.0)
    parser.add_argument("--prelift-contact-timeout-s", type=float, default=2.0)
    parser.add_argument("--execute-steps", type=int, default=10)
    parser.add_argument("--binary-gripper", action=argparse.BooleanOptionalAction, default=False,
                        help="Threshold gripper predictions into Z1 open (-0.90) / close (0.0) targets")
    parser.add_argument("--gripper-binary-threshold-rad", type=float, default=-0.75,
                        help="Close when the predicted angle is >= this threshold; default -0.75 rad")
    parser.add_argument("--gripper-action-encoding", choices=("angle", "binary"), default="angle",
                        help="binary: unnormalized policy output >= 0.5 closes, otherwise opens")
    parser.add_argument("--stop-on-pregrasp-contact", action="store_true")
    parser.add_argument(
        "--freeze-target-until-close", action=argparse.BooleanOptionalAction,
        default=False,
        help="Hold the target fixed until the policy first commands gripper closing",
    )
    parser.add_argument("--max-joint-step", type=float, default=0.10)
    parser.add_argument("--stale-timeout", type=float, default=0.5)
    parser.add_argument("--input-sync-tolerance", type=float, default=0.025,
                        help="Maximum spread of model image/state sensor times, seconds")
    parser.add_argument("--input-sync-max-age", type=float, default=0.2,
                        help="Maximum age of a synchronized model observation in simulation seconds")
    parser.add_argument("--input-sync-timeout", type=float, default=0.5,
                        help="Wall-clock wait for a fresh synchronized observation before stopping")
    parser.add_argument("--tracking-error", type=float, default=0.08)
    parser.add_argument("--tracking-error-duration", type=float, default=0.5)
    parser.add_argument("--sim-clock-timeout", type=float, default=10.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--preview-episodes", type=int, default=3)
    parser.add_argument(
        "--training-manifest", type=Path, default=None
    )
    parser.add_argument("--training-dataset-root", type=Path)
    parser.add_argument("--max-training-target-distance-m", type=float)
    parser.add_argument("--reject-blank-cameras", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("/home/lp531/smolvla_eval")
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.gripper_min_close_hold_s) or args.gripper_min_close_hold_s < 0:
        parser.error("--gripper-min-close-hold-s must be finite and nonnegative")
    if args.gripper_min_close_hold_s > 0 and not args.binary_gripper:
        parser.error("close hold requires --binary-gripper")
    if args.gripper_action_encoding == "binary" and not args.binary_gripper:
        parser.error("binary action encoding requires --binary-gripper")
    if not np.isfinite(args.gripper_binary_threshold_rad) or not -0.90 < args.gripper_binary_threshold_rad < 0.0:
        parser.error("--gripper-binary-threshold-rad must be between -0.90 and 0.0")
    if args.policy_type == "diffusion" and (args.objects != ["cup"] or args.execute_steps > 15):
        parser.error("This Diffusion checkpoint requires cup and execute-steps <= 15")
    if args.policy_type == "pi05" and args.checkpoint is None:
        parser.error("PI05 evaluation requires an explicit LoRA checkpoint")
    if args.target_xy_bounds is not None:
        x_min, x_max, y_min, y_max = args.target_xy_bounds
        if not (np.isfinite(args.target_xy_bounds).all()
                and x_min < x_max and y_min < y_max):
            parser.error("--target-xy-bounds must be finite increasing intervals")
        args.target_xy_bounds = tuple(args.target_xy_bounds)
    if args.policy_type in {"act", "diffusion"}:
        if args.checkpoint is None or args.objects not in (["beverage_bottle"], ["apple"], ["cup"]):
            parser.error("ACT/Diffusion scene evaluation requires a checkpoint and a single cup, apple or beverage_bottle target")
        if args.target_xy_bounds is None:
            parser.error("ACT/Diffusion scene evaluation requires --target-xy-bounds")
    if args.checkpoint is None:
        args.checkpoint = latest_checkpoint()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.episodes_per_object <= 0 or args.execute_steps <= 0:
        parser.error("episode and execution counts must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in (
        args.input_sync_tolerance, args.input_sync_max_age, args.input_sync_timeout,
    )) or args.input_sync_tolerance > args.input_sync_max_age:
        parser.error("input sync requires finite positive values and tolerance <= max age")
    if args.max_training_target_distance_m is not None and (
        not np.isfinite(args.max_training_target_distance_m)
        or args.max_training_target_distance_m <= 0
        or args.training_dataset_root is None
    ):
        parser.error("--max-training-target-distance-m requires a positive distance and --training-dataset-root")
    if args.eval_seeds is not None and (
        args.policy_type not in {"act", "pi05", "diffusion"}
        or len(args.eval_seeds) != args.episodes_per_object
        or len(set(args.eval_seeds)) != len(args.eval_seeds)
        or any(seed < 0 for seed in args.eval_seeds)
    ):
        parser.error("--eval-seeds requires one unique nonnegative seed per ACT, Diffusion or PI05 episode")
    if (
        args.timeout <= 0 or args.rate <= 0 or args.stale_timeout <= 0
        or args.tracking_error <= 0 or args.tracking_error_duration <= 0
        or args.sim_clock_timeout <= 0
        or args.startup_timeout <= 0
    ):
        parser.error("timeouts and rate must be positive")
    execution_rate = args.rate if args.execution_rate is None else args.execution_rate
    if (not np.isfinite(execution_rate) or execution_rate < args.rate
            or not np.isclose(execution_rate / args.rate, round(execution_rate / args.rate))):
        parser.error("--execution-rate must be a positive integer multiple of --rate")
    if (not np.isfinite(args.prelift_contact_stable_s) or args.prelift_contact_stable_s < 0
            or not np.isfinite(args.prelift_contact_timeout_s)
            or args.prelift_contact_timeout_s <= args.prelift_contact_stable_s):
        parser.error("contact timeout must exceed the nonnegative stable duration")
    return args


def main(args=None) -> None:
    cli_args = rclpy.utilities.remove_ros_args(args=args)[1:]
    parsed = parse_args(cli_args)
    rclpy.init(args=args)
    node = SmolVLASimRollout(parsed)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        output = node.run_suite()
        node.get_logger().info(f"Evaluation report written to {output}")
    except KeyboardInterrupt:
        pass
    finally:
        node._set_active(False)
        executor.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
