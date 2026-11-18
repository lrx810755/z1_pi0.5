"""HTTP client for the ROS-side SmolVLA evaluator GUI bridge."""

import json
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import SMOLVLA_GUI_BRIDGE_URL


class SmolVLAEvaluatorClient:
    def __init__(self, base_url=SMOLVLA_GUI_BRIDGE_URL, timeout_s=0.2):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._cache = {"connected": False, "phase": "GUI_BRIDGE_OFFLINE"}
        self._cache_time = 0.0

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            if now - self._cache_time < 0.1:
                return dict(self._cache)
        try:
            data = self._request("/api/status")
        except RuntimeError as exc:
            data = {
                "connected": False,
                "phase": "GUI_BRIDGE_OFFLINE",
                "error": str(exc),
                "available_objects": [
                    "cup",
                    "apple",
                    "peach",
                    "mouse",
                    "beverage_bottle",
                ],
            }
        with self._lock:
            self._cache = data
            self._cache_time = now
            return dict(data)

    def set_target(self, target_object):
        return self._request("/api/target", {"target_object": target_object})

    def reset(self):
        return self._request("/api/reset", {})

    def generate_pregrasp(self):
        return self._request("/api/pregrasp/generate", {})

    def execute_pregrasp(self):
        return self._request("/api/pregrasp/execute", {})

    def _request(self, path, payload=None):
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                message = json.loads(exc.read().decode("utf-8")).get(
                    "error", str(exc)
                )
            except Exception:
                message = str(exc)
            raise RuntimeError(message) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"SmolVLA GUI bridge unavailable: {exc}"
            ) from exc
