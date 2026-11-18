import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import (
    ARM_JOINTS,
    REAL_ARM_BRIDGE_PATH,
    REAL_ARM_CONTROLLER_BIN,
    REAL_ARM_CONTROLLER_DIR,
    REAL_ARM_ENABLED,
    REAL_ARM_HOST,
    REAL_ARM_JOINT_SPEED_RAD_S,
    REAL_ARM_MANAGE_BRIDGE,
    REAL_ARM_MANAGE_CONTROLLER,
    REAL_ARM_PORT,
    REAL_ARM_PYTHON,
    REAL_ARM_REQUEST_TIMEOUT_S,
    REAL_ARM_ROBOT_IP,
    REAL_ARM_SDK_DIR,
    REAL_ARM_STARTUP_TIMEOUT_S,
    REAL_ARM_SYNC_INTERVAL_S,
    REAL_ARM_URL,
    REAL_ARM_WATCHDOG_S,
)


class RealArmClient:
    def __init__(self):
        self.enabled = bool(REAL_ARM_ENABLED)
        self.base_url = REAL_ARM_URL.rstrip("/")
        self.timeout_s = REAL_ARM_REQUEST_TIMEOUT_S
        self.sync_interval_s = REAL_ARM_SYNC_INTERVAL_S
        self.controller_proc = None
        self.bridge_proc = None
        self.started = False
        self.last_sync_time = 0.0
        self.last_error = ""
        self.last_status = self._disabled_status("真机未启用；设置 Z1_REAL_ARM_ENABLED=1 后才会连接")

    def start(self):
        if not self.enabled:
            return self.last_status
        if self.started:
            return self.status()
        self.started = True
        try:
            if REAL_ARM_MANAGE_CONTROLLER:
                self._start_controller()
            if REAL_ARM_MANAGE_BRIDGE and not self._bridge_reachable():
                self._start_bridge()
            self._wait_for_bridge()
            self.last_status = self.status()
        except Exception as exc:
            self.last_error = str(exc)
            self.last_status = self._error_status(self.last_error)
        return self.last_status

    def close(self):
        if self.enabled:
            try:
                self.stop()
            except Exception:
                pass
        for proc in (self.bridge_proc, self.controller_proc):
            self._terminate_process(proc)

    def status(self):
        if not self.enabled:
            self.last_status = self._disabled_status("真机未启用；设置 Z1_REAL_ARM_ENABLED=1 后才会连接")
            return dict(self.last_status)
        try:
            self.last_status = self._get("/status")
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
            self.last_status = self._error_status(self.last_error)
        return dict(self.last_status)

    def enable(self, targets_by_name, speed_scale):
        if not self.enabled:
            return self.last_status
        return self._post_command(
            "/enable",
            {
                "targets": self._targets_payload(targets_by_name),
                "speed_scale": float(speed_scale),
            },
        )

    def disable(self):
        if not self.enabled:
            return self.last_status
        return self._post_command("/disable", {})

    def hold(self):
        if not self.enabled:
            return self.last_status
        return self._post_command("/hold", {})

    def stop(self):
        if not self.enabled:
            return self.last_status
        return self._post_command("/stop", {})

    def send_targets(self, targets_by_name, speed_scale, force=False):
        if not self.enabled:
            return self.last_status
        now = time.monotonic()
        if not force and now - self.last_sync_time < self.sync_interval_s:
            return self.last_status
        self.last_sync_time = now
        return self._post_command(
            "/set_targets",
            {
                "targets": self._targets_payload(targets_by_name),
                "speed_scale": float(speed_scale),
            },
        )

    def _start_controller(self):
        binary = Path(REAL_ARM_CONTROLLER_BIN)
        if not binary.exists():
            raise FileNotFoundError(f"z1_ctrl 不存在: {binary}")
        log = open("/tmp/z1_real_arm_controller.log", "ab", buffering=0)
        self.controller_proc = subprocess.Popen(
            [str(binary)],
            cwd=str(Path(REAL_ARM_CONTROLLER_DIR) / "build"),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=self._process_env(),
            start_new_session=True,
        )
        time.sleep(0.8)
        if self.controller_proc.poll() is not None:
            raise RuntimeError("z1_ctrl 启动后立即退出，请查看 /tmp/z1_real_arm_controller.log")

    def _start_bridge(self):
        bridge = Path(REAL_ARM_BRIDGE_PATH)
        if not bridge.exists():
            raise FileNotFoundError(f"真机 bridge 脚本不存在: {bridge}")
        log = open("/tmp/z1_real_arm_bridge.log", "ab", buffering=0)
        self.bridge_proc = subprocess.Popen(
            [REAL_ARM_PYTHON, str(bridge)],
            cwd=str(bridge.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=self._process_env(),
            start_new_session=True,
        )

    def _wait_for_bridge(self):
        deadline = time.monotonic() + REAL_ARM_STARTUP_TIMEOUT_S
        last_error = ""
        while time.monotonic() < deadline:
            if self.bridge_proc is not None and self.bridge_proc.poll() is not None:
                raise RuntimeError("真机 bridge 启动后立即退出，请查看 /tmp/z1_real_arm_bridge.log")
            try:
                self._get("/status")
                return
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.2)
        raise TimeoutError(f"等待真机 bridge 超时: {last_error}")

    def _bridge_reachable(self):
        try:
            self._get("/status")
            return True
        except Exception:
            return False

    def _post_command(self, path, payload):
        try:
            self.last_status = self._post(path, payload)
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
            self.last_status = self._error_status(self.last_error)
        return dict(self.last_status)

    def _get(self, path):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error = json.loads(exc.read().decode("utf-8")).get("error")
            except Exception:
                error = str(exc)
            raise RuntimeError(error or str(exc)) from exc

    def _targets_payload(self, targets_by_name):
        payload = {}
        for name in ARM_JOINTS:
            if name in targets_by_name:
                payload[name] = float(targets_by_name[name])
        return payload

    def _process_env(self):
        env = os.environ.copy()
        sdk_lib = str(Path(REAL_ARM_SDK_DIR) / "lib")
        controller_lib = str(Path(REAL_ARM_CONTROLLER_DIR) / "lib")
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(item for item in (sdk_lib, controller_lib, existing_ld) if item)
        env["PYTHONPATH"] = ":".join(item for item in (sdk_lib, env.get("PYTHONPATH", "")) if item)
        env["Z1_REAL_ARM_HOST"] = REAL_ARM_HOST
        env["Z1_REAL_ARM_PORT"] = str(REAL_ARM_PORT)
        env["Z1_REAL_ARM_IP"] = REAL_ARM_ROBOT_IP
        env["Z1_REAL_ARM_SDK_DIR"] = str(REAL_ARM_SDK_DIR)
        env["Z1_REAL_ARM_WATCHDOG_S"] = str(REAL_ARM_WATCHDOG_S)
        env["Z1_REAL_ARM_JOINT_SPEED_RAD_S"] = str(REAL_ARM_JOINT_SPEED_RAD_S)
        return env

    def _terminate_process(self, proc):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=2.0)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass

    def _disabled_status(self, message):
        return {
            "enabled": False,
            "configured": False,
            "connected": False,
            "tracking": False,
            "estop": False,
            "message": message,
            "q": None,
            "gripper_q": None,
        }

    def _error_status(self, message):
        return {
            "enabled": False,
            "configured": True,
            "connected": False,
            "tracking": False,
            "estop": False,
            "message": message,
            "q": None,
            "gripper_q": None,
        }
