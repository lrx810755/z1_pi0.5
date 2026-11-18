import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .config import DEFAULT_SPEED_SCALE, FRONTEND_PATH, HOST, MAX_JSON_BODY_BYTES, PORT


class TeleopHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_file(FRONTEND_PATH, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            data = self.server.teleop_state.snapshot()
            if self.server.real_arm is not None:
                data["real_arm"] = self.server.real_arm.status()
            if self.server.red_ball_manager is not None:
                data["red_ball"] = self.server.red_ball_manager.snapshot()
            if self.server.red_ball_grasp is not None:
                data["grasp"] = self.server.red_ball_grasp.snapshot()
            if self.server.fixed_demo is not None:
                data["fixed_demo"] = self.server.fixed_demo.snapshot()
            if self.server.smolvla_client is not None:
                data["smolvla"] = self.server.smolvla_client.snapshot()
            self.send_json(data)
            return
        if path == "/api/camera.bmp":
            frame, frame_time, error = self.server.camera_frame_store.snapshot()
            if frame is None:
                self.send_error_json(503, error)
                return
            self.send_binary(frame, "image/bmp", extra_headers={"X-Frame-Time": f"{frame_time:.3f}"})
            return
        self.send_error_json(404, "not found")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self.read_json()
            if path == "/api/enable":
                data = self.server.teleop_state.set_enabled(True, body.get("speed_scale", DEFAULT_SPEED_SCALE))
            elif path == "/api/stop":
                self.cancel_grasp("急停触发，抓取任务已停止")
                self.cancel_fixed_demo("急停触发，固定 demo 已停止")
                data = self.server.teleop_state.estop_now()
            elif path == "/api/hold":
                self.cancel_grasp("保持当前位置，抓取任务已停止")
                self.cancel_fixed_demo("保持当前位置，固定 demo 已停止")
                data = self.server.teleop_state.hold()
            elif path == "/api/home":
                self.cancel_grasp("回安全位，抓取任务已停止")
                self.cancel_fixed_demo("回安全位，固定 demo 已停止")
                data = self.server.teleop_state.home(body.get("speed_scale", DEFAULT_SPEED_SCALE))
            elif path == "/api/speed":
                data = self.server.teleop_state.set_speed(body.get("speed_scale", DEFAULT_SPEED_SCALE))
            elif path == "/api/set_joint":
                self.cancel_grasp_after_detection("手动关节控制，抓取任务已停止")
                self.cancel_fixed_demo("手动关节控制，固定 demo 已停止")
                data = self.server.teleop_state.set_joint(body["joint"], body["value"], body.get("speed_scale", DEFAULT_SPEED_SCALE))
            elif path == "/api/jog":
                self.cancel_grasp_after_detection("手动步进控制，抓取任务已停止")
                self.cancel_fixed_demo("手动步进控制，固定 demo 已停止")
                data = self.server.teleop_state.jog(body["joint"], body["delta"], body.get("speed_scale", DEFAULT_SPEED_SCALE))
            elif path == "/api/grasp_red_ball":
                self.cancel_fixed_demo("视觉抓取启动，固定 demo 已停止")
                if self.server.red_ball_grasp is None:
                    raise RuntimeError("红球抓取控制器尚未初始化")
                data = self.server.red_ball_grasp.start(body.get("speed_scale", DEFAULT_SPEED_SCALE))
            elif path == "/api/grasp_stop":
                if self.server.red_ball_grasp is None:
                    raise RuntimeError("红球抓取控制器尚未初始化")
                data = self.server.red_ball_grasp.cancel()
            elif path == "/api/fixed_demo_grasp":
                self.cancel_grasp("固定 demo 启动，视觉抓取任务已停止")
                if self.server.fixed_demo is None:
                    raise RuntimeError("固定 demo 控制器尚未初始化")
                data = self.server.fixed_demo.start(body.get("speed_scale", DEFAULT_SPEED_SCALE))
            elif path == "/api/fixed_demo_stop":
                if self.server.fixed_demo is None:
                    raise RuntimeError("固定 demo 控制器尚未初始化")
                data = self.server.fixed_demo.cancel()
            elif path == "/api/reset_red_ball":
                self.cancel_grasp("红球已重置，抓取任务已停止")
                self.cancel_fixed_demo("红球已重置，固定 demo 已停止")
                if self.server.red_ball_manager is None:
                    raise RuntimeError("红球管理器尚未初始化")
                position = body.get("position_xyz_m", body.get("position"))
                ball = self.server.red_ball_manager.request_reset(position)
                data = self.server.teleop_state.snapshot()
                data["red_ball"] = ball
                if self.server.red_ball_grasp is not None:
                    data["grasp"] = self.server.red_ball_grasp.snapshot()
            elif path == "/api/smolvla_target":
                if self.server.smolvla_client is None:
                    raise RuntimeError("SmolVLA 判定桥尚未初始化")
                data = self.server.smolvla_client.set_target(body.get("target_object", ""))
            elif path == "/api/smolvla_reset":
                if self.server.smolvla_client is None:
                    raise RuntimeError("SmolVLA 判定桥尚未初始化")
                data = self.server.smolvla_client.reset()
            elif path == "/api/smolvla_pregrasp_generate":
                if self.server.smolvla_client is None:
                    raise RuntimeError("SmolVLA 判定桥尚未初始化")
                data = self.server.smolvla_client.generate_pregrasp()
            elif path == "/api/smolvla_pregrasp_execute":
                if self.server.smolvla_client is None:
                    raise RuntimeError("SmolVLA 判定桥尚未初始化")
                data = self.server.smolvla_client.execute_pregrasp()
            else:
                self.send_error_json(404, "not found")
                return
            self.send_json(data)
        except Exception as exc:
            self.send_error_json(400, str(exc))

    def cancel_grasp(self, message):
        if self.server.red_ball_grasp is not None:
            self.server.red_ball_grasp.cancel(message)

    def cancel_fixed_demo(self, message):
        if self.server.fixed_demo is not None:
            self.server.fixed_demo.cancel(message)

    def cancel_grasp_after_detection(self, message):
        if self.server.red_ball_grasp is None:
            return
        state = self.server.red_ball_grasp.snapshot()
        if state.get("active") and state.get("phase") == "detect":
            return
        self.server.red_ball_grasp.cancel(message)

    def read_json(self):
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError(f"无效 Content-Length: {raw_length!r}") from exc
        if length > MAX_JSON_BODY_BYTES:
            raise ValueError(f"JSON 请求体过大: {length} bytes")
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def send_file(self, path, content_type):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.write_response_body(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.write_response_body(data)

    def send_binary(self, payload, content_type, status=200, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.write_response_body(payload)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status=status)

    def write_response_body(self, payload):
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class TeleopHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_http_server(
    teleop_state,
    camera_frame_store,
    red_ball_manager=None,
    red_ball_grasp=None,
    trajectory_executor=None,
    fixed_demo=None,
    real_arm=None,
    smolvla_client=None,
    host=HOST,
    port=PORT,
):
    server = TeleopHTTPServer((host, port), TeleopHandler)
    server.teleop_state = teleop_state
    server.camera_frame_store = camera_frame_store
    server.red_ball_manager = red_ball_manager
    server.red_ball_grasp = red_ball_grasp
    server.trajectory_executor = trajectory_executor
    server.fixed_demo = fixed_demo
    server.real_arm = real_arm
    server.smolvla_client = smolvla_client
    thread = threading.Thread(target=server.serve_forever, name="z1-teleop-http", daemon=True)
    thread.start()
    return server
