import json
import threading
import urllib.error
import urllib.request

import numpy as np

from .config import DEFAULT_SPEED_SCALE, MOVEIT_BRIDGE_TIMEOUT_S, MOVEIT_BRIDGE_URL


def wxyz_to_xyzw(quat):
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return [float(x), float(y), float(z), float(w)]



class MoveItBridge:
    def __init__(self, url=MOVEIT_BRIDGE_URL, timeout_s=MOVEIT_BRIDGE_TIMEOUT_S):
        self.url = url
        self.timeout_s = timeout_s
        self.lock = threading.Lock()
        self.request_id = 0
        self.active = False
        self.phase = "idle"
        self.result = None
        self.error = ""

    def plan_async(self, phase, position_xyz, orientation_wxyz, speed_scale=DEFAULT_SPEED_SCALE, start_joint_state=None):
        with self.lock:
            if self.active:
                return False
            self.request_id += 1
            request_id = self.request_id
            self.active = True
            self.phase = phase
            self.result = None
            self.error = ""

        payload = {
            "phase": phase,
            "group": "z1_arm",
            "end_effector_link": "link06",
            "position_xyz_m": np.asarray(position_xyz, dtype=np.float64).tolist(),
            "orientation_xyzw": wxyz_to_xyzw(orientation_wxyz),
            "speed_scale": float(speed_scale),
        }
        if start_joint_state is not None:
            payload["start_joint_state"] = start_joint_state
        thread = threading.Thread(
            target=self._worker,
            args=(request_id, payload),
            name=f"z1-moveit-{phase}",
            daemon=True,
        )
        thread.start()
        return True

    def _worker(self, request_id, payload):
        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
            result = json.loads(raw) if raw else {}
            if not result.get("success", False):
                raise RuntimeError(result.get("error") or "MoveIt 规划或执行失败")
            self._finish(request_id, result, "")
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = ""
            message = f"HTTP {exc.code}: {body or exc.reason}"
            self._finish(request_id, None, message)
        except Exception as exc:
            self._finish(request_id, None, str(exc))

    def _finish(self, request_id, result, error):
        with self.lock:
            if request_id != self.request_id:
                return
            self.active = False
            self.result = result
            self.error = error

    def cancel(self, error="MoveIt 请求已取消"):
        with self.lock:
            self.request_id += 1
            self.active = False
            self.phase = "idle"
            self.result = None
            self.error = error

    def snapshot(self):
        with self.lock:
            return {
                "active": self.active,
                "phase": self.phase,
                "url": self.url,
                "result": self.result,
                "error": self.error,
            }
