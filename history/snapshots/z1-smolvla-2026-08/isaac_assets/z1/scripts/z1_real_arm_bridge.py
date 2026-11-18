#!/usr/bin/env python3
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper"]
HOST = os.environ.get("Z1_REAL_ARM_HOST", "127.0.0.1")
PORT = int(os.environ.get("Z1_REAL_ARM_PORT", "12001"))
SDK_DIR = Path(
    os.environ.get(
        "Z1_REAL_ARM_SDK_DIR", Path.home() / "unitree_z1" / "z1_sdk"
    )
)
WATCHDOG_S = float(os.environ.get("Z1_REAL_ARM_WATCHDOG_S", "0.45"))
DEFAULT_JOINT_SPEED = float(os.environ.get("Z1_REAL_ARM_JOINT_SPEED_RAD_S", "0.6"))
MAX_JOINT_SPEED = float(os.environ.get("Z1_REAL_ARM_MAX_JOINT_SPEED_RAD_S", "1.2"))
POSITION_DEADBAND_RAD = float(os.environ.get("Z1_REAL_ARM_DEADBAND_RAD", "0.006"))


class RealArmBridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.arm = None
        self.arm_state = None
        self.enabled = False
        self.estop = False
        self.connected = False
        self.tracking = False
        self.target = None
        self.speed_scale = 0.3
        self.message = "等待连接真实 Z1"
        self.last_command_time = 0.0
        self.last_state_time = 0.0
        self.stop_event = threading.Event()
        self.control_thread = threading.Thread(target=self._control_loop, name="z1-real-arm-control", daemon=True)
        self.control_thread.start()

    def connect(self):
        with self.lock:
            if self.arm is not None:
                return
        sdk_lib = SDK_DIR / "lib"
        if str(sdk_lib) not in sys.path:
            sys.path.insert(0, str(sdk_lib))
        import unitree_arm_interface

        arm = unitree_arm_interface.ArmInterface(hasGripper=True)
        arm.loopOn()
        with self.lock:
            self.arm = arm
            self.arm_state = unitree_arm_interface.ArmFSMState
            self.connected = True
            self.message = "真实 Z1 bridge 已连接 SDK"

    def close(self):
        self.stop_event.set()
        with self.lock:
            arm = self.arm
        if arm is not None:
            try:
                arm.jointCtrlCmd([0.0] * 7, 0.0)
            except Exception:
                pass
            try:
                arm.loopOff()
            except Exception:
                pass

    def enable(self, targets, speed_scale):
        self.connect()
        target = self._target_from_payload(targets)
        with self.lock:
            self.arm.startTrack(self.arm_state.JOINTCTRL)
            self.target = target
            self.speed_scale = self._safe_speed_scale(speed_scale)
            self.enabled = True
            self.estop = False
            self.tracking = True
            self.last_command_time = time.monotonic()
            self.message = "真实 Z1 已使能，正在镜像仿真目标"
        return self.snapshot()

    def set_targets(self, targets, speed_scale):
        self.connect()
        target = self._target_from_payload(targets)
        with self.lock:
            self.target = target
            self.speed_scale = self._safe_speed_scale(speed_scale)
            self.last_command_time = time.monotonic()
            if not self.estop:
                self.enabled = True
                self.tracking = True
            self.message = "真实 Z1 目标已更新"
        return self.snapshot()

    def hold(self, message="真实 Z1 保持当前位置"):
        with self.lock:
            self.enabled = False
            self.tracking = False
            self.target = self._read_positions_locked()
            self.message = message
        self._send_zero()
        return self.snapshot()

    def disable(self):
        return self.hold("真实 Z1 已禁用")

    def stop(self):
        with self.lock:
            self.enabled = False
            self.tracking = False
            self.estop = True
            self.target = self._read_positions_locked()
            self.message = "真实 Z1 急停已触发"
            arm = self.arm
            arm_state = self.arm_state
        self._send_zero()
        if arm is not None and arm_state is not None:
            try:
                arm.setFsm(arm_state.PASSIVE)
            except Exception:
                pass
        return self.snapshot()

    def home(self, speed_scale):
        self.connect()
        with self.lock:
            self.enabled = False
            self.tracking = False
            self.speed_scale = self._safe_speed_scale(speed_scale)
            self.message = "真实 Z1 正在执行 backToStart"
            arm = self.arm
        arm.backToStart()
        with self.lock:
            self.target = self._read_positions_locked()
            self.enabled = False
            self.tracking = False
            self.message = "真实 Z1 已回到控制器起始位"
        return self.snapshot()

    def snapshot(self):
        with self.lock:
            try:
                q7 = self._read_positions_locked()
            except Exception as exc:
                q7 = None
                self.connected = False
                self.tracking = False
                self.message = f"真实 Z1 状态读取失败: {exc}"
            state = None
            if self.arm is not None:
                try:
                    state = int(self.arm.getCurrentState())
                except Exception:
                    state = None
            return {
                "enabled": bool(self.enabled),
                "configured": True,
                "connected": bool(self.connected),
                "tracking": bool(self.tracking),
                "estop": bool(self.estop),
                "message": self.message,
                "q": None if q7 is None else q7[:6],
                "gripper_q": None if q7 is None else q7[6],
                "target": None if self.target is None else list(self.target),
                "speed_scale": self.speed_scale,
                "fsm_state": state,
                "last_command_age_s": None if self.last_command_time <= 0 else time.monotonic() - self.last_command_time,
                "last_state_time": self.last_state_time,
            }

    def _control_loop(self):
        while not self.stop_event.is_set():
            sleep_s = 0.01
            try:
                with self.lock:
                    arm = self.arm
                    enabled = self.enabled
                    estop = self.estop
                    target = None if self.target is None else np.asarray(self.target, dtype=np.float64)
                    speed = min(MAX_JOINT_SPEED, DEFAULT_JOINT_SPEED * max(0.05, min(1.0, self.speed_scale)))
                    stale = self.last_command_time > 0 and time.monotonic() - self.last_command_time > WATCHDOG_S
                    if stale and self.enabled:
                        self.enabled = False
                        self.tracking = False
                        self.message = "真实 Z1 命令超时，已停止"
                if arm is None:
                    time.sleep(0.1)
                    continue
                if not enabled or estop or target is None or stale:
                    arm.jointCtrlCmd([0.0] * 7, 0.0)
                    time.sleep(sleep_s)
                    continue
                current = np.asarray(self._read_positions(), dtype=np.float64)
                error = target - current
                directions = np.zeros(7, dtype=np.float64)
                active = np.abs(error) > POSITION_DEADBAND_RAD
                directions[active] = np.clip(error[active] / max(speed * 0.08, 1e-6), -1.0, 1.0)
                arm.jointCtrlCmd(directions.tolist(), speed)
            except Exception as exc:
                with self.lock:
                    self.connected = False
                    self.tracking = False
                    self.enabled = False
                    self.message = f"真实 Z1 控制循环错误: {exc}"
                time.sleep(0.1)
            time.sleep(sleep_s)

    def _read_positions_locked(self):
        if self.arm is None:
            return None
        return self._read_positions()

    def _read_positions(self):
        q = np.asarray(self.arm.lowstate.getQ(), dtype=np.float64).reshape(-1)
        gripper_q = float(self.arm.lowstate.getGripperQ())
        self.last_state_time = time.time()
        return np.concatenate([q[:6], np.asarray([gripper_q], dtype=np.float64)]).tolist()

    def _target_from_payload(self, targets):
        current = self._read_positions()
        values = list(current)
        for idx, name in enumerate(ARM_JOINTS):
            if name in targets:
                values[idx] = float(targets[name])
        return values

    def _send_zero(self):
        with self.lock:
            arm = self.arm
        if arm is not None:
            try:
                arm.jointCtrlCmd([0.0] * 7, 0.0)
            except Exception:
                pass

    def _safe_speed_scale(self, speed_scale):
        try:
            return max(0.05, min(1.0, float(speed_scale)))
        except Exception:
            return 0.3


bridge = RealArmBridge()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/status", "/api/status"):
            self.send_json(bridge.snapshot())
            return
        self.send_error_json(404, "not found")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self.read_json()
            if path in ("/enable", "/api/enable"):
                data = bridge.enable(body.get("targets", {}), body.get("speed_scale", 0.3))
            elif path in ("/set_targets", "/api/set_targets"):
                data = bridge.set_targets(body.get("targets", {}), body.get("speed_scale", 0.3))
            elif path in ("/hold", "/api/hold"):
                data = bridge.hold()
            elif path in ("/disable", "/api/disable"):
                data = bridge.disable()
            elif path in ("/stop", "/api/stop"):
                data = bridge.stop()
            elif path in ("/home", "/api/home"):
                data = bridge.home(body.get("speed_scale", 0.3))
            else:
                self.send_error_json(404, "not found")
                return
            self.send_json(data)
        except Exception as exc:
            self.send_error_json(400, str(exc))

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status=status)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)

    def shutdown(signum, frame):
        threading.Thread(target=server.shutdown, name="z1-real-arm-http-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Z1 real arm bridge listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
