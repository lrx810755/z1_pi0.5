"""Coordinate a complete, one-click SmolVLA episode reset."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


CURRICULUM_LEVELS = ("easy", "medium", "full")


class EpisodeReset(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_episode_reset")
        self._reset_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._recording = False
        group = ReentrantCallbackGroup()

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._status_pub = self.create_publisher(
            String, "/smolvla/episode_reset_status", qos
        )
        self.create_subscription(
            String,
            "/smolvla/recording_status",
            self._recording_callback,
            10,
            callback_group=group,
        )
        self._trigger_clients = {
            "stop_assistant": self.create_client(
                Trigger, "/smolvla/stop_assistant", callback_group=group
            ),
            "stop_motion": self.create_client(
                Trigger, "/smolvla/stop_motion", callback_group=group
            ),
            "discard": self.create_client(
                Trigger, "/smolvla/recording/discard", callback_group=group
            ),
            "reset_scene": self.create_client(
                Trigger, "/smolvla/reset_scene", callback_group=group
            ),
            "randomize_scene": self.create_client(
                Trigger, "/smolvla/randomize_scene", callback_group=group
            ),
            "validate_randomized_scene": self.create_client(
                Trigger,
                "/smolvla/validate_randomized_scene",
                callback_group=group,
            ),
            "reset_evaluator": self.create_client(
                Trigger, "/smolvla/reset_evaluator", callback_group=group
            ),
            "reset_assistant": self.create_client(
                Trigger, "/smolvla/reset_assistant", callback_group=group
            ),
            "reload_collisions": self.create_client(
                Trigger,
                "/smolvla/reload_object_collisions",
                callback_group=group,
            ),
        }
        for level in CURRICULUM_LEVELS:
            self._trigger_clients[f"randomize_scene_{level}"] = (
                self.create_client(
                    Trigger,
                    f"/smolvla/randomize_scene_{level}",
                    callback_group=group,
                )
            )
        self.create_service(
            Trigger,
            "/smolvla/reset_episode",
            self._reset_callback,
            callback_group=group,
        )
        self.create_service(
            Trigger,
            "/smolvla/randomize_episode",
            self._randomize_callback,
            callback_group=group,
        )
        self.create_service(
            Trigger,
            "/smolvla/randomize_episode_easy",
            self._randomize_easy_callback,
            callback_group=group,
        )
        self.create_service(
            Trigger,
            "/smolvla/randomize_episode_medium",
            self._randomize_medium_callback,
            callback_group=group,
        )
        self.create_service(
            Trigger,
            "/smolvla/randomize_episode_full",
            self._randomize_full_callback,
            callback_group=group,
        )
        self._publish_status("READY", "一键 Episode 复位已就绪")
        self.get_logger().info(
            "Episode reset ready: /smolvla/reset_episode and "
            "/smolvla/randomize_episode"
        )

    def _recording_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            return
        with self._state_lock:
            self._recording = bool(status.get("recording", False))

    def _publish_status(self, phase: str, message: str, **details: Any) -> None:
        payload = String()
        payload.data = json.dumps(
            {
                "phase": phase,
                "ready": phase != "WAITING",
                "resetting": phase == "RESETTING",
                "message": message,
                **details,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        self._status_pub.publish(payload)

    def _call(self, key: str, required: bool, timeout_s: float = 4.0) -> str:
        client = self._trigger_clients[key]
        if not client.wait_for_service(timeout_sec=0.5):
            if required:
                raise RuntimeError(f"{client.srv_name} 服务不可用")
            return "服务未启动，已跳过"
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(f"{client.srv_name} 超时")
        response = future.result()
        if response is None or not response.success:
            message = response.message if response is not None else "无响应"
            if required:
                raise RuntimeError(f"{client.srv_name}: {message}")
            return f"已跳过：{message}"
        return response.message

    def _reset_callback(self, _request, response):
        return self._run_reset(response, randomize_level=None)

    def _randomize_callback(self, _request, response):
        return self._run_reset(response, randomize_level="easy")

    def _randomize_easy_callback(self, _request, response):
        return self._run_reset(response, randomize_level="easy")

    def _randomize_medium_callback(self, _request, response):
        return self._run_reset(response, randomize_level="medium")

    def _randomize_full_callback(self, _request, response):
        return self._run_reset(response, randomize_level="full")

    def _run_reset(self, response, randomize_level):
        if not self._reset_lock.acquire(blocking=False):
            response.success = False
            response.message = "Episode 正在复位，请稍候"
            return response
        completed = []
        randomization_message = ""
        try:
            self._publish_status("RESETTING", "正在停止当前轨迹", step="stop")
            completed.append(self._call("stop_assistant", required=False))
            completed.append(self._call("stop_motion", required=True))

            with self._state_lock:
                recording = self._recording
            if recording:
                self._publish_status(
                    "RESETTING", "正在丢弃未保存录制", step="discard"
                )
                completed.append(self._call("discard", required=False))

            self._publish_status(
                "RESETTING", "正在恢复 Isaac 场景", step="scene"
            )
            completed.append(self._call("reset_scene", required=True))
            if randomize_level is not None:
                self._publish_status(
                    "RESETTING",
                    f"正在生成 {randomize_level} 课程随机桌面布局",
                    step="randomize",
                    curriculum_level=randomize_level,
                )
                randomization_message = self._call(
                    f"randomize_scene_{randomize_level}",
                    required=True,
                    timeout_s=8.0,
                )
                completed.append(randomization_message)
                # Let the newly dynamic objects acquire table contact before
                # validating their settled poses and copying them into
                # MoveIt's planning scene.
                time.sleep(1.0)
                self._publish_status(
                    "RESETTING",
                    "正在验收物体稳定性、桌面接触和碰撞穿透",
                    step="validate_randomization",
                    curriculum_level=randomize_level,
                )
                randomization_message = self._call(
                    "validate_randomized_scene",
                    required=True,
                    timeout_s=4.0,
                )
                completed.append(randomization_message)

            self._publish_status(
                "RESETTING", "正在清除判定与规划状态", step="state"
            )
            completed.append(self._call("reset_evaluator", required=True))
            completed.append(self._call("reset_assistant", required=False))
            # The evaluator reset makes grasp_success false. Reloading then
            # detaches any held collision and restores all five world objects.
            completed.append(self._call("reload_collisions", required=False))

            message = (
                "Episode 已随机化并完整复位，可以开始下一轮；"
                f"{randomization_message}"
                if randomize_level is not None
                else "Episode 已完整复位，可以直接开始下一轮"
            )
            self._publish_status(
                "COMPLETE",
                message,
                completed_steps=len(completed),
                curriculum_level=randomize_level,
            )
            response.success = True
            response.message = message
        except Exception as exc:
            message = f"Episode 复位失败：{exc}"
            self.get_logger().error(message)
            self._publish_status("ERROR", message)
            response.success = False
            response.message = message
        finally:
            self._reset_lock.release()
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeReset()
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
