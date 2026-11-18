"""Expose the SmolVLA grasp evaluator through a small local web GUI."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List
from urllib.parse import urlparse

from ament_index_python.packages import get_package_share_path
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .grasp_evaluator import OBJECT_FRAME_NAMES


MAX_JSON_BODY_BYTES = 16 * 1024
GUIDED_GRASP_TIMEOUT_S = 600.0
GUIDED_PLACE_TIMEOUT_S = 600.0
LAYOUT_RESAMPLE_ATTEMPTS = 3
GRID_LAYOUT_RESAMPLE_ATTEMPTS = 10
CURRICULUM_LEVELS = ("easy", "medium", "full")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    @property
    def bridge(self) -> "GraspGuiBridge":
        return self.server.bridge  # type: ignore[attr-defined, no-any-return]

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_bytes(
                self.bridge.frontend.read_bytes(), "text/html; charset=utf-8"
            )
        elif path == "/api/status":
            self._send_json(self.bridge.snapshot())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            body = self._read_json()
            path = urlparse(self.path).path
            if path == "/api/target":
                result = self.bridge.set_target(
                    str(body.get("target_object", ""))
                )
            elif path == "/api/reset":
                result = self.bridge.reset_evaluator()
            elif path == "/api/reset_episode":
                result = self.bridge.reset_episode()
            elif path == "/api/force_stop":
                result = self.bridge.force_stop()
            elif path == "/api/randomize_episode":
                result = self.bridge.randomize_episode(
                    str(body.get("curriculum_level", "easy")),
                    body.get("target_grid"),
                )
            elif path == "/api/demonstration/auto":
                result = self.bridge.auto_demonstration(
                    str(body.get("target_object", "")),
                    str(body.get("curriculum_level", "easy")),
                    body.get("target_grid"),
                )
            elif path == "/api/recording/start":
                result = self.bridge.start_recording()
            elif path == "/api/recording/save":
                result = self.bridge.save_recording()
            elif path == "/api/recording/discard":
                result = self.bridge.discard_recording()
            else:
                self._send_json({"error": "not found"}, status=404)
                return
            self._send_json(result)
        except (ValueError, RuntimeError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length > MAX_JSON_BODY_BYTES:
            raise ValueError("JSON request body is too large")
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_bytes(
        self, payload: bytes, content_type: str, status: int = 200
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class GraspGuiBridge(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_grasp_gui_bridge")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8767)
        self.declare_parameter("status_timeout_s", 5.0)

        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.status_timeout = float(
            self.get_parameter("status_timeout_s").value
        )
        self.frontend = (
            Path(get_package_share_path("z1_isaac_bridge"))
            / "web"
            / "smolvla_task_gui.html"
        )
        if not self.frontend.is_file():
            raise FileNotFoundError(
                f"GUI file does not exist: {self.frontend}"
            )

        self._lock = threading.Lock()
        self._auto_lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "phase": "WAITING_FOR_STATE",
            "grasp_success": False,
            "task_success": False,
            "target_object": "apple",
            "failure_reason": "",
        }
        self._last_status_monotonic = 0.0
        self._pregrasp_status: Dict[str, Any] = {
            "phase": "IDLE",
            "message": "等待预抓取助手",
            "plan_ready": False,
        }
        self._collision_scene_status: Dict[str, Any] = {
            "phase": "WAITING",
            "ready": False,
            "message": "等待碰撞场景同步节点",
        }
        self._object_collision_status: Dict[str, Any] = {
            "phase": "WAITING",
            "ready": False,
            "message": "等待动态物体碰撞同步节点",
        }
        self._recording_status: Dict[str, Any] = {
            "phase": "WAITING",
            "ready": False,
            "recording": False,
            "message": "等待训练集录制节点",
        }
        self._episode_reset_status: Dict[str, Any] = {
            "phase": "WAITING",
            "ready": False,
            "resetting": False,
            "message": "等待 Episode 复位节点",
        }
        self._scene_randomization_status: Dict[str, Any] = {}
        self._auto_status: Dict[str, Any] = {
            "phase": "IDLE",
            "running": False,
            "step": "",
            "message": "选择目标后可开始一键自动示教",
        }
        self._force_stop_event = threading.Event()
        self._force_stop_status: Dict[str, Any] = {
            "phase": "IDLE",
            "running": False,
            "message": "强制停止未触发",
        }
        callback_group = ReentrantCallbackGroup()
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._target_pub = self.create_publisher(
            String, "/smolvla/target_object", 10
        )
        self._randomization_command_pub = self.create_publisher(
            String, "/smolvla/scene_randomization_command", 10
        )
        self.create_subscription(
            String,
            "/smolvla/status",
            self._status_callback,
            10,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/pregrasp_status",
            self._pregrasp_status_callback,
            latched_qos,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/planning_scene_status",
            self._collision_scene_status_callback,
            latched_qos,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/object_collision_status",
            self._object_collision_status_callback,
            latched_qos,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/recording_status",
            self._recording_status_callback,
            latched_qos,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/episode_reset_status",
            self._episode_reset_status_callback,
            latched_qos,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/scene_reset_status",
            self._scene_randomization_status_callback,
            latched_qos,
            callback_group=callback_group,
        )
        self._reset_client = self.create_client(
            Trigger,
            "/smolvla/reset_evaluator",
            callback_group=callback_group,
        )
        self._episode_reset_client = self.create_client(
            Trigger,
            "/smolvla/reset_episode",
            callback_group=callback_group,
        )
        self._episode_randomize_client = self.create_client(
            Trigger,
            "/smolvla/randomize_episode",
            callback_group=callback_group,
        )
        self._episode_randomize_clients = {
            level: self.create_client(
                Trigger,
                f"/smolvla/randomize_episode_{level}",
                callback_group=callback_group,
            )
            for level in CURRICULUM_LEVELS
        }
        self._generate_client = self.create_client(
            Trigger,
            "/smolvla/auto_generate_pregrasp",
            callback_group=callback_group,
        )
        self._execute_guided_client = self.create_client(
            Trigger,
            "/smolvla/auto_execute_pregrasp",
            callback_group=callback_group,
        )
        self._guided_grasp_client = self.create_client(
            Trigger,
            "/smolvla/auto_guided_grasp",
            callback_group=callback_group,
        )
        self._guided_place_client = self.create_client(
            Trigger,
            "/smolvla/auto_guided_place",
            callback_group=callback_group,
        )
        self._stop_assistant_client = self.create_client(
            Trigger,
            "/smolvla/stop_assistant",
            callback_group=callback_group,
        )
        self._stop_motion_client = self.create_client(
            Trigger,
            "/smolvla/stop_motion",
            callback_group=callback_group,
        )
        self._recording_start_client = self.create_client(
            Trigger,
            "/smolvla/recording/start",
            callback_group=callback_group,
        )
        self._recording_save_client = self.create_client(
            Trigger,
            "/smolvla/recording/save",
            callback_group=callback_group,
        )
        self._recording_discard_client = self.create_client(
            Trigger,
            "/smolvla/recording/discard",
            callback_group=callback_group,
        )

        self._server = _Server((self.host, self.port), _Handler)
        self._server.bridge = self  # type: ignore[attr-defined]
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="smolvla-grasp-gui-http",
            daemon=True,
        )
        self._server_thread.start()
        self.get_logger().info(
            f"SmolVLA grasp GUI: http://{self.host}:{self.port}"
        )

    def _status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(
                f"Ignoring malformed /smolvla/status: {exc}"
            )
            return
        with self._lock:
            self._status = status
            self._last_status_monotonic = time.monotonic()

    def _pregrasp_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(
                f"Ignoring malformed /smolvla/pregrasp_status: {exc}"
            )
            return
        with self._lock:
            self._pregrasp_status = status

    def _collision_scene_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(
                f"Ignoring malformed /smolvla/planning_scene_status: {exc}"
            )
            return
        with self._lock:
            self._collision_scene_status = status

    def _object_collision_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(
                f"Ignoring malformed /smolvla/object_collision_status: {exc}"
            )
            return
        with self._lock:
            self._object_collision_status = status

    def _recording_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(
                f"Ignoring malformed /smolvla/recording_status: {exc}"
            )
            return
        with self._lock:
            self._recording_status = status

    def _scene_randomization_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._scene_randomization_status = status

    def _episode_reset_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(
                f"Ignoring malformed /smolvla/episode_reset_status: {exc}"
            )
            return
        with self._lock:
            self._episode_reset_status = status
            if status.get("phase") == "COMPLETE" and not self._auto_lock.locked():
                self._auto_status = {
                    "phase": "IDLE", "running": False, "step": "已复位",
                    "message": "Episode 已完整复位，可以开始下一轮",
                }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            status = dict(self._status)
            last_update = self._last_status_monotonic
            collision_scene = dict(self._collision_scene_status)
            object_collisions = dict(self._object_collision_status)
            recording = dict(self._recording_status)
            episode_reset = dict(self._episode_reset_status)
            auto = dict(self._auto_status)
            force_stop = dict(self._force_stop_status)
        age = time.monotonic() - last_update if last_update else None
        status.update(
            {
                "connected": age is not None and age <= self.status_timeout,
                "status_age_s": round(age, 3) if age is not None else None,
                "available_objects": list(OBJECT_FRAME_NAMES),
                "curriculum_levels": list(CURRICULUM_LEVELS),
                "collision_scene": collision_scene,
                "object_collisions": object_collisions,
                "recording": recording,
                "episode_reset": episode_reset,
                "auto_demonstration": auto,
                "force_stop": force_stop,
            }
        )
        return status

    def set_target(self, requested: str) -> Dict[str, Any]:
        target = requested.strip().lower()
        if target not in OBJECT_FRAME_NAMES:
            raise ValueError(f"unknown target_object {target!r}")
        message = String()
        message.data = target
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self._target_pub.publish(message)
            time.sleep(0.05)
            with self._lock:
                evaluator_target = self._status.get("target_object")
                assistant_target = self._pregrasp_status.get("target_object")
            if evaluator_target == target and assistant_target == target:
                break
        else:
            raise RuntimeError(
                "目标切换未被判定器和预抓取助手共同确认，请重试"
            )
        return {
            "success": True,
            "message": f"目标已切换为 {target}",
            "target_object": target,
        }

    def reset_evaluator(self) -> Dict[str, Any]:
        return self._call_trigger(
            self._reset_client,
            "/smolvla/reset_evaluator",
            timeout_s=10.0,
            ignore_force_stop=True,
        )

    def reset_episode(self) -> Dict[str, Any]:
        if self._auto_lock.locked():
            raise RuntimeError("抓取停止流程尚未退出，请稍后再重置 Episode")
        result = self._call_trigger(
            self._episode_reset_client,
            "/smolvla/reset_episode",
            timeout_s=30.0,
            ignore_force_stop=True,
        )
        if not result.get("success", False):
            return result
        self._force_stop_event.clear()
        with self._lock:
            self._auto_status = {
                "phase": "IDLE", "running": False, "step": "已复位",
                "message": "Episode 已完整复位，可以开始下一轮",
            }
            self._force_stop_status = {
                "phase": "IDLE",
                "running": False,
                "message": "Episode 已重置，强制停止已解除",
            }
        return result

    def force_stop(self) -> Dict[str, Any]:
        """Cancel the workflow and latch both controllers at measured state."""
        self._force_stop_event.set()
        with self._lock:
            self._force_stop_status = {
                "phase": "STOPPING",
                "running": True,
                "message": "正在取消抓取并锁定当前关节位置",
            }
        messages = []
        errors = []
        for client, service_name in (
            (self._stop_assistant_client, "/smolvla/stop_assistant"),
            (self._stop_motion_client, "/smolvla/stop_motion"),
        ):
            try:
                result = self._call_trigger(
                    client,
                    service_name,
                    timeout_s=5.0,
                    ignore_force_stop=True,
                )
                messages.append(result["message"])
            except RuntimeError as exc:
                errors.append(f"{service_name}: {exc}")
        phase = "STOPPED" if not errors else "STOP_ERROR"
        message = "；".join(messages)
        if errors:
            message = "；".join(filter(None, (message, *errors)))
        with self._lock:
            self._force_stop_status = {
                "phase": phase,
                "running": False,
                "message": message,
            }
            self._auto_status = {
                "phase": "STOPPED",
                "running": False,
                "step": "强制停止",
                "message": message,
            }
        if errors:
            raise RuntimeError(message)
        return {"success": True, "message": message, "phase": phase}

    @staticmethod
    def _normalize_curriculum_level(level: str) -> str:
        normalized = level.strip().lower()
        if normalized not in CURRICULUM_LEVELS:
            raise ValueError(
                f"unknown curriculum_level {normalized!r}; "
                f"expected one of {CURRICULUM_LEVELS}"
            )
        return normalized

    @staticmethod
    def _normalize_target_grid(target_grid: Any) -> tuple[int, int] | None:
        if target_grid is None:
            return None
        if not isinstance(target_grid, (list, tuple)) or len(target_grid) != 2:
            raise ValueError("target_grid must be [column, row]")
        if any(isinstance(value, bool) for value in target_grid):
            raise ValueError("target_grid values must be integers")
        try:
            column, row = (int(value) for value in target_grid)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_grid values must be integers") from exc
        if list(target_grid) != [column, row]:
            raise ValueError("target_grid values must be integers")
        if not 0 <= column < 5 or not 0 <= row < 5:
            raise ValueError("target_grid is outside the configured 5x5 grid")
        return column, row

    def randomize_episode(
        self,
        curriculum_level: str = "easy",
        target_grid: Any = None,
    ) -> Dict[str, Any]:
        level = self._normalize_curriculum_level(curriculum_level)
        grid = self._normalize_target_grid(target_grid)
        if grid is not None and level != "full":
            raise ValueError("target_grid can only be used with full curriculum")
        command = String()
        command.data = json.dumps({"target_grid": grid})
        expected_grid = list(grid) if grid is not None else None
        with self._lock:
            self._scene_randomization_status = {}
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self._randomization_command_pub.publish(command)
            time.sleep(0.05)
            with self._lock:
                status = dict(self._scene_randomization_status)
            if (
                status.get("phase") == "GRID_CONFIGURED"
                and status.get("target_grid") == expected_grid
            ):
                break
        else:
            raise RuntimeError("Isaac 未确认目标采样方格设置")
        return self._call_trigger(
            self._episode_randomize_clients[level],
            f"/smolvla/randomize_episode_{level}",
            timeout_s=40.0,
        )

    def _generate_pregrasp_for_auto(self) -> Dict[str, Any]:
        return self._call_trigger(
            self._generate_client,
            "/smolvla/auto_generate_pregrasp",
            timeout_s=120.0,
        )

    def _execute_pregrasp_for_auto(self) -> Dict[str, Any]:
        return self._call_trigger(
            self._execute_guided_client,
            "/smolvla/auto_execute_pregrasp",
            timeout_s=120.0,
        )

    def auto_demonstration(
        self,
        requested_target: str,
        curriculum_level: str = "easy",
        target_grid: Any = None,
    ) -> Dict[str, Any]:
        if not self._auto_lock.acquire(blocking=False):
            raise RuntimeError("一键自动示教正在运行，请勿重复点击")
        self._force_stop_event.clear()
        with self._lock:
            self._force_stop_status = {
                "phase": "ARMED",
                "running": False,
                "message": "强制停止可用",
            }
        recording_started = False
        try:
            with self._lock:
                if bool(self._recording_status.get("recording", False)):
                    raise RuntimeError("已有 Episode 正在录制，请先保存或丢弃")
                if not bool(self._recording_status.get("ready", False)):
                    raise RuntimeError("训练集录制节点尚未就绪")
                if not bool(self._episode_reset_status.get("ready", False)):
                    raise RuntimeError("Episode 复位节点尚未就绪")
                if not bool(self._collision_scene_status.get("ready", False)):
                    raise RuntimeError("MoveIt 静态碰撞场景尚未就绪")
                if not bool(self._object_collision_status.get("ready", False)):
                    raise RuntimeError("动态物体碰撞场景尚未就绪")
                status_age = time.monotonic() - self._last_status_monotonic
                if status_age > self.status_timeout:
                    raise RuntimeError("抓取判定状态已离线或过期")

            self._set_auto_status("RESETTING", "准备场景", "正在切换目标")
            self.set_target(requested_target)
            level = self._normalize_curriculum_level(curriculum_level)
            grid = self._normalize_target_grid(target_grid)
            # Keep resampling layouts until one passes; a blocked corridor or
            # an unreachable candidate is a layout problem, not a reason to
            # abandon the episode.  Only the operator force-stop ends the loop.
            layout_errors: List[str] = []
            layout_attempt = 0
            while True:
                if self._force_stop_event.is_set():
                    raise RuntimeError("用户强制停止抓取")
                layout_attempt += 1
                if layout_attempt <= LAYOUT_RESAMPLE_ATTEMPTS:
                    attempt_label = f"布局 {layout_attempt}"
                else:
                    attempt_label = (
                        f"布局 {layout_attempt}"
                        f"（已连续失败 {layout_attempt - 1} 次，继续重采样）"
                    )
                self._set_auto_status(
                    "RESETTING",
                    "准备场景",
                    f"正在按 {level} 课程等级随机化并完整复位"
                    f"（{attempt_label}）",
                )
                self.randomize_episode(level, grid)
                if self._force_stop_event.is_set():
                    raise RuntimeError("用户强制停止抓取")
                self._set_auto_status(
                    "PLANNING",
                    "规划预抓取",
                    "正在验证夹爪扫掠通道并生成碰撞安全轨迹"
                    f"（{attempt_label}）",
                )
                try:
                    self._generate_pregrasp_for_auto()
                    break
                except RuntimeError as exc:
                    if self._force_stop_event.is_set():
                        raise RuntimeError("用户强制停止抓取") from exc
                    layout_errors.append(str(exc))
                    del layout_errors[:-LAYOUT_RESAMPLE_ATTEMPTS]
                    if (
                        grid is not None
                        and layout_attempt >= GRID_LAYOUT_RESAMPLE_ATTEMPTS
                    ):
                        raise RuntimeError(
                            "GRID_UNREACHABLE："
                            f"锁定方格 {grid} 连续 "
                            f"{GRID_LAYOUT_RESAMPLE_ATTEMPTS} 个布局均不可达；"
                            f"last_error={layout_errors[-1]}"
                        ) from exc

            self._set_auto_status(
                "EXECUTING", "执行预抓取", "正在开始录制并移动"
            )
            # Planning can leave the last held command older than the recorder's
            # causal-action window. Establish a fresh, real hold before frame 1.
            self._call_trigger(
                self._stop_motion_client, "/smolvla/stop_motion", timeout_s=5.0
            )
            self.start_recording()
            recording_started = True
            self._execute_pregrasp_for_auto()

            self._set_auto_status(
                "GRASPING", "抓取并抬升", "正在对准、闭爪和验证抓取"
            )
            self._guided_grasp_for_auto()

            self._set_auto_status(
                "PLACING", "搬运并放置", "正在安全搬运到蓝盒"
            )
            self._guided_place_for_auto()

            self._set_auto_status(
                "SAVING", "保存 Episode", "任务成功，正在写入训练集"
            )
            saved = self.save_recording()
            recording_started = False
            message = f"一键示教成功；{saved['message']}"
            self._set_auto_status("SUCCESS", "完成", message, running=False)
            return {"success": True, "message": message}
        except Exception as exc:
            cleanup_messages = []
            if recording_started:
                try:
                    self.discard_recording()
                    cleanup_messages.append("已丢弃失败录制")
                except RuntimeError as cleanup_exc:
                    cleanup_messages.append(f"丢弃录制失败：{cleanup_exc}")
            cleanup_messages.append(
                "场景保持在失败位置；请手动点击“一键重置 Episode”归位"
            )
            suffix = "；".join(cleanup_messages)
            forced = self._force_stop_event.is_set()
            message = (
                f"一键示教已强制停止：{exc}"
                if forced
                else f"一键示教失败：{exc}"
            )
            if suffix:
                message = f"{message}（{suffix}）"
            self._set_auto_status(
                "STOPPED" if forced else "ERROR",
                "强制停止" if forced else "失败",
                message,
                running=False,
            )
            raise RuntimeError(message) from exc
        finally:
            self._auto_lock.release()

    def _set_auto_status(
        self,
        phase: str,
        step: str,
        message: str,
        running: bool = True,
    ) -> None:
        with self._lock:
            self._auto_status = {
                "phase": phase,
                "running": running,
                "step": step,
                "message": message,
            }

    def _guided_grasp_for_auto(self) -> Dict[str, Any]:
        return self._call_trigger(
            self._guided_grasp_client,
            "/smolvla/auto_guided_grasp",
            # Slow simulation can need well over one wall-clock minute for
            # the segmented 220 mm physical lift. A Trigger call cannot be
            # cancelled here, so an early client timeout would discard the
            # recording while the successful physical grasp keeps running.
            timeout_s=GUIDED_GRASP_TIMEOUT_S,
        )

    def _guided_place_for_auto(self) -> Dict[str, Any]:
        return self._call_trigger(
            self._guided_place_client,
            "/smolvla/auto_guided_place",
            # Placement also contains several planned motions and needs
            # headroom when Isaac Sim is running below real time.
            timeout_s=GUIDED_PLACE_TIMEOUT_S,
        )

    def start_recording(self) -> Dict[str, Any]:
        return self._call_trigger(
            self._recording_start_client,
            "/smolvla/recording/start",
            timeout_s=10.0,
        )

    def save_recording(self) -> Dict[str, Any]:
        return self._call_trigger(
            self._recording_save_client,
            "/smolvla/recording/save",
            timeout_s=30.0,
        )

    def discard_recording(self) -> Dict[str, Any]:
        return self._call_trigger(
            self._recording_discard_client,
            "/smolvla/recording/discard",
            timeout_s=30.0,
            ignore_force_stop=True,
        )

    def _call_trigger(
        self,
        client,
        service_name: str,
        timeout_s: float,
        ignore_force_stop: bool = False,
    ) -> Dict[str, Any]:
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"{service_name} service is unavailable")
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            if self._force_stop_event.is_set() and not ignore_force_stop:
                raise RuntimeError("用户强制停止抓取")
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(f"{service_name} timed out")
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                response.message
                if response is not None
                else f"{service_name} failed"
            )
        return {"success": True, "message": response.message}

    def destroy_node(self) -> bool:
        self._server.shutdown()
        self._server.server_close()
        self._server_thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspGuiBridge()
    executor = MultiThreadedExecutor(num_threads=2)
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
