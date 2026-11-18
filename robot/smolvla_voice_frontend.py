#!/usr/bin/env python3
"""Terminal and microphone frontend for the SmolVLA voice backend."""

from __future__ import annotations

import argparse
import json
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
import speech_recognition as sr
from std_msgs.msg import String

from asr import (
    RuntimeConfig,
    VoiceTranslator,
    clean_command,
    suppress_stderr,
    validate_microphone,
)
from z1_isaac_bridge.smolvla_voice_core import match_target


class VoiceFrontend(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_voice_frontend")
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        command_qos = QoSProfile(depth=10)
        command_qos.reliability = ReliabilityPolicy.RELIABLE
        command_qos.durability = DurabilityPolicy.VOLATILE
        self._publisher = self.create_publisher(
            String, "/smolvla/voice_command", command_qos
        )
        self.create_subscription(
            String, "/smolvla/voice_status", self._status_callback, qos
        )
        self._condition = threading.Condition()
        self.status: dict = {}
        self._revision = 0

    def _status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._condition:
            self.status = status
            self._revision += 1
            self._condition.notify_all()
        state = status.get("state", "UNKNOWN")
        detail = status.get("error") or status.get("message", "")
        print(f"\n[后端] {state}: {detail}", flush=True)

    @property
    def state(self) -> str:
        with self._condition:
            return str(self.status.get("state", ""))

    @property
    def revision(self) -> int:
        with self._condition:
            return self._revision

    def wait_for(self, states: set[str], timeout: float | None = None) -> str:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while str(self.status.get("state", "")) not in states:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"backend did not reach {sorted(states)}")
                self._condition.wait(timeout=remaining)
            return str(self.status["state"])

    def wait_for_update(self, revision: int, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._revision <= revision:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("backend did not acknowledge command")
                self._condition.wait(timeout=remaining)
            return str(self.status.get("state", ""))

    def publish(self, command: str, target: str | None = None) -> None:
        message = String()
        payload = {"command": command}
        if target is not None:
            payload["target"] = target
        message.data = json.dumps(payload, ensure_ascii=False)
        self._publisher.publish(message)


def listen_once(
    translator: VoiceTranslator,
    phrase_time_limit: int,
    timeout: float | None,
) -> str:
    try:
        with suppress_stderr(not translator.config.show_alsa_logs):
            microphone = sr.Microphone(device_index=translator.config.mic_index)
            with microphone as source:
                translator.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                translator.recognizer.energy_threshold += 150
                audio = translator.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
    except sr.WaitTimeoutError:
        return ""
    except Exception as exc:
        print(f"[错误] 麦克风采集失败: {exc}")
        return ""
    return translator.recognize_speech(audio)


def parse_voice_command(translator: VoiceTranslator, text: str):
    cleaned = clean_command(text.replace(translator.wake_word, ""))
    if len(cleaned) < 2:
        return None
    intent = translator.process_command(cleaned)
    action = str(intent.get("action", "none")).lower()
    if action == "stop":
        return "stop", None
    if action not in {"pick", "grab", "grasp", "take"}:
        print("[提醒] 只接受抓取或停止指令")
        return None
    target = match_target(cleaned, str(intent.get("object", "")))
    if target is None:
        print("[提醒] 目标必须明确为杯子、苹果、桃子或饮料瓶")
        return None
    return "grasp", target


def run_terminal(frontend: VoiceFrontend, translator: VoiceTranslator) -> None:
    frontend.wait_for({"IDLE"}, timeout=300.0)
    while rclpy.ok():
        terminal = input("\n输入 start 随机化场景，或输入 quit 退出: ").strip().lower()
        if terminal in {"quit", "exit", "q"}:
            return
        if terminal != "start":
            print("[提醒] 当前只接受 start 或 quit")
            continue
        revision = frontend.revision
        frontend.publish("start")
        state = frontend.wait_for_update(revision, timeout=10.0)
        if state == "RANDOMIZING":
            state = frontend.wait_for({"LISTENING", "IDLE"}, timeout=180.0)
        if state != "LISTENING":
            continue

        while frontend.state == "LISTENING":
            print("[聆听] 请说“抓取杯子/苹果/桃子/饮料瓶”，也可带唤醒词。")
            spoken = listen_once(translator, phrase_time_limit=6, timeout=None)
            if not spoken:
                continue
            print(f"[识别文本] {spoken}")
            parsed = parse_voice_command(translator, spoken)
            if parsed is None:
                continue
            command, target = parsed
            frontend.publish(command, target)
            if command == "stop":
                frontend.wait_for({"IDLE"}, timeout=10.0)
                break
            frontend.wait_for({"EXECUTING", "IDLE"}, timeout=10.0)

        while frontend.state == "EXECUTING":
            spoken = listen_once(translator, phrase_time_limit=3, timeout=1.0)
            if not spoken:
                continue
            parsed = parse_voice_command(translator, spoken)
            if parsed and parsed[0] == "stop":
                print("[停止] 已向策略后端发送停止请求")
                frontend.publish("stop")
        if frontend.state != "IDLE":
            frontend.wait_for({"IDLE"}, timeout=30.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmolVLA 语音抓取前端")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--llm-model", default="qwen2.5:3b")
    parser.add_argument("--wake-word", default="小白")
    parser.add_argument("--mic-index", type=int)
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--show-alsa-logs", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not validate_microphone(args.mic_index, args.show_alsa_logs):
        return 1
    config = RuntimeConfig(
        whisper_model=args.whisper_model,
        llm_model=args.llm_model,
        wake_word=args.wake_word,
        mic_index=args.mic_index,
        no_ollama=args.no_ollama,
        show_alsa_logs=args.show_alsa_logs,
    )
    translator = VoiceTranslator(config)
    rclpy.init()
    frontend = VoiceFrontend()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(frontend)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        run_terminal(frontend, translator)
    except (KeyboardInterrupt, EOFError):
        if frontend.state in {"LISTENING", "EXECUTING"}:
            frontend.publish("stop")
    finally:
        executor.shutdown()
        thread.join(timeout=2.0)
        frontend.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
