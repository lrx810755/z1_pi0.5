"""Interactive ROS backend for voice-controlled SmolVLA rollouts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from queue import Empty, Queue
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .smolvla_sim_rollout import SmolVLASimRollout, parse_args
from .smolvla_voice_core import (
    InvalidTransition,
    VoiceCommand,
    VoiceSession,
    decode_command,
)


class SmolVLAVoiceBackend(SmolVLASimRollout):
    def __init__(self, args) -> None:
        super().__init__(args)
        self._commands: Queue[VoiceCommand] = Queue()
        self._session = VoiceSession()
        self._session_lock = threading.Lock()
        self._stop_event = threading.Event()
        status_qos = QoSProfile(depth=10)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        status_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        command_qos = QoSProfile(depth=10)
        command_qos.reliability = ReliabilityPolicy.RELIABLE
        command_qos.durability = DurabilityPolicy.VOLATILE
        self._voice_status_pub = self.create_publisher(
            String, "/smolvla/voice_status", status_qos
        )
        self.create_subscription(
            String, "/smolvla/voice_command", self._command_callback, command_qos
        )

    def _command_callback(self, message: String) -> None:
        try:
            command = decode_command(message.data)
        except ValueError as exc:
            self._publish_status(error=str(exc))
            return
        with self._session_lock:
            state = self._session.state
        if command.command == "stop" and state == "EXECUTING":
            self._stop_event.set()
            self._publish_status(message="operator stop requested")
            return
        self._commands.put(command)

    def _publish_status(self, state: str | None = None, **details) -> None:
        with self._session_lock:
            payload = {
                "state": state or self._session.state,
                "seed": self._session.seed,
                "target": self._session.target,
                **details,
            }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._voice_status_pub.publish(message)
        self.get_logger().info(message.data)

    def run_voice_loop(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = self.args.output / f"smolvla_voice_{stamp}"
        root.mkdir(parents=True, exist_ok=False)
        episode_number = 0
        self._publish_status(message="type start in the voice terminal")
        while rclpy.ok():
            try:
                command = self._commands.get(timeout=0.2)
            except Empty:
                continue
            try:
                with self._session_lock:
                    action = self._session.accept(command)
                if action == "randomize":
                    self._publish_status(message="FULL scene randomization started")
                    try:
                        seed = self._randomize_scene()
                    except Exception as exc:
                        with self._session_lock:
                            self._session.randomization_failed()
                        self._publish_status(error=str(exc))
                        continue
                    with self._session_lock:
                        self._session.randomized(seed)
                    self._publish_status(message="scene ready; waiting for voice command")
                elif action == "cancel":
                    self._set_active(False)
                    self._publish_status(message="prepared scene cancelled")
                elif action == "execute":
                    episode_number += 1
                    with self._session_lock:
                        target = self._session.target
                        seed = self._session.seed
                    self._stop_event.clear()
                    self._publish_status(message="SmolVLA grasp started")
                    result = self.run_episode(
                        str(target), episode_number, root,
                        prepared_seed=seed, stop_event=self._stop_event,
                    )
                    terminal = "FINISHED" if result["success"] else "FAULT"
                    self._publish_status(state=terminal, result=result)
                    with self._session_lock:
                        self._session.episode_done()
                    self._publish_status(
                        message="attempt ended; type start for the next scene",
                        last_result={
                            "success": result["success"],
                            "failure_category": result["failure_category"],
                            "target_object": result["target_object"],
                        },
                    )
            except InvalidTransition as exc:
                self._publish_status(error=str(exc))
            except Exception as exc:
                self._set_active(False)
                with self._session_lock:
                    if self._session.state == "EXECUTING":
                        self._session.episode_done()
                    elif self._session.state == "RANDOMIZING":
                        self._session.randomization_failed()
                self._publish_status(error=f"backend error: {exc}")
        return root


def main(args=None) -> None:
    cli_args = rclpy.utilities.remove_ros_args(args=args)[1:]
    parsed = parse_args(cli_args)
    rclpy.init(args=args)
    node = SmolVLAVoiceBackend(parsed)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        output = node.run_voice_loop()
        node.get_logger().info(f"Voice rollout records written to {output}")
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._set_active(False)
        executor.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
