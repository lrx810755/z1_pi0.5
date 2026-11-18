"""Run automatic demonstrations until the requested episodes are saved."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TARGET_OBJECTS = (
    "cup",
    "apple",
    "peach",
    "mouse",
    "beverage_bottle",
)
CURRICULUM_LEVELS = ("easy", "medium", "full")
TARGET_GRID_SHAPE = (5, 5)


class ApiError(RuntimeError):
    """The local training API rejected or could not complete a request."""


class JsonApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get(self, path: str, timeout_s: float = 10.0) -> Dict[str, Any]:
        return self._request("GET", path, None, timeout_s)

    def post(
        self,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        return self._request("POST", path, payload or {}, timeout_s)

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]],
        timeout_s: float,
    ) -> Dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout_s) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                message = body
            raise ApiError(f"HTTP {exc.code}: {message}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ApiError(str(exc)) from exc
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiError(f"API returned invalid JSON: {body[:200]}") from exc
        if not isinstance(result, dict):
            raise ApiError("API returned a non-object JSON response")
        return result


class ContinuousTrainingRunner:
    """Retry complete demonstrations and count only successfully saved data."""

    def __init__(
        self,
        api: JsonApiClient,
        target: str,
        curriculum: str,
        episodes: int,
        target_grid: Optional[tuple[int, int]] = None,
        retry_delay_s: float = 2.0,
        max_attempts: int = 0,
        max_failures: int = 0,
        output: Callable[[str], None] = print,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if target not in TARGET_OBJECTS:
            raise ValueError(f"unknown target {target!r}")
        if curriculum not in CURRICULUM_LEVELS:
            raise ValueError(f"unknown curriculum {curriculum!r}")
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        if retry_delay_s < 0.0:
            raise ValueError("retry_delay_s cannot be negative")
        if max_attempts < 0:
            raise ValueError("max_attempts cannot be negative")
        if max_failures < 0:
            raise ValueError("max_failures cannot be negative")
        if target_grid is not None:
            if curriculum != "full":
                raise ValueError("target_grid requires full curriculum")
            column, row = target_grid
            if not (
                0 <= column < TARGET_GRID_SHAPE[0]
                and 0 <= row < TARGET_GRID_SHAPE[1]
            ):
                raise ValueError(
                    f"target_grid {target_grid} is outside {TARGET_GRID_SHAPE}"
                )
        self.api = api
        self.target = target
        self.curriculum = curriculum
        self.episodes = episodes
        self.target_grid = target_grid
        self.retry_delay_s = retry_delay_s
        self.max_attempts = max_attempts
        self.max_failures = max_failures
        self.output = output
        self.sleep = sleep
        self.successes = 0
        self.failures = 0
        self.attempts = 0
        self._stopping = False

    def run(self) -> int:
        self._wait_for_stack()
        grid_label = (
            f", grid={self.target_grid}" if self.target_grid is not None else ""
        )
        self.output(
            f"开始连续采集：target={self.target}, "
            f"curriculum={self.curriculum}, successes={self.episodes}"
            f"{grid_label}"
        )
        while self.successes < self.episodes and not self._stopping:
            if self.max_failures and self.failures >= self.max_failures:
                self.output(
                    f"当前组合累计失败 {self.failures} 次，"
                    "已复位并交由调度器跳过"
                )
                return 4
            if self.max_attempts and self.attempts >= self.max_attempts:
                raise RuntimeError(
                    f"达到最大尝试次数 {self.max_attempts}；"
                    f"已保存 {self.successes}/{self.episodes} 轮"
                )
            self.attempts += 1
            self.output(
                f"[尝试 {self.attempts}] 开始；"
                f"已保存 {self.successes}/{self.episodes}"
            )
            try:
                payload = {
                    "target_object": self.target,
                    "curriculum_level": self.curriculum,
                }
                if self.target_grid is not None:
                    payload["target_grid"] = list(self.target_grid)
                result = self.api.post(
                    "/api/demonstration/auto",
                    payload,
                    # Layout resampling no longer gives up on its own, so
                    # the client timeout only guards a dead HTTP server.
                    timeout_s=7200.0,
                )
                if not result.get("success", False):
                    raise ApiError(result.get("message", "自动示教未成功"))
            except KeyboardInterrupt:
                self._stopping = True
                break
            except Exception as exc:
                self.failures += 1
                self.output(f"[尝试 {self.attempts}] 失败：{exc}")
                self._recover_failed_episode()
                if "GRID_UNREACHABLE" in str(exc):
                    self.output(
                        "当前物体与方格已确认不可达；"
                        "返回采集调度器并切换下一组合"
                    )
                    return 3
                continue

            self.successes += 1
            self.output(
                f"[尝试 {self.attempts}] 保存成功；"
                f"进度 {self.successes}/{self.episodes}"
            )
            if self.successes < self.episodes and self.retry_delay_s:
                self.sleep(self.retry_delay_s)

        if self._stopping:
            self._stop_and_discard()
            return 130
        self.output(
            f"采集完成：保存 {self.successes} 轮，失败并丢弃 "
            f"{self.failures} 轮，总尝试 {self.attempts} 次"
        )
        return 0

    def request_stop(self) -> None:
        self._stopping = True

    def _wait_for_stack(self) -> None:
        while not self._stopping:
            try:
                status = self.api.get("/api/status", timeout_s=5.0)
                recording = status.get("recording", {})
                reset = status.get("episode_reset", {})
                collisions = status.get("collision_scene", {})
                objects = status.get("object_collisions", {})
                if (
                    status.get("connected", False)
                    and recording.get("ready", False)
                    and reset.get("ready", False)
                    and collisions.get("ready", False)
                    and objects.get("ready", False)
                ):
                    return
                reason = "训练栈尚未全部就绪"
            except Exception as exc:
                reason = str(exc)
            self.output(f"等待训练栈：{reason}")
            self.sleep(max(1.0, self.retry_delay_s))
        raise KeyboardInterrupt

    def _recover_failed_episode(self) -> None:
        """Never start another attempt until a complete reset succeeds."""
        self._best_effort_post("/api/force_stop", "强制停止")
        self._best_effort_post("/api/recording/discard", "丢弃失败录制")
        while not self._stopping:
            try:
                result = self.api.post(
                    "/api/reset_episode", {}, timeout_s=120.0
                )
                if not result.get("success", False):
                    raise ApiError(result.get("message", "Episode 复位失败"))
                self.output("失败 Episode 已丢弃并完整复位")
                if self.retry_delay_s:
                    self.sleep(self.retry_delay_s)
                return
            except Exception as exc:
                self.output(f"Episode 复位失败，将继续重试：{exc}")
                self.sleep(max(1.0, self.retry_delay_s))

    def _best_effort_post(self, path: str, label: str) -> None:
        try:
            self.api.post(path, {}, timeout_s=30.0)
        except Exception as exc:
            self.output(f"{label}未完成（复位流程将继续处理）：{exc}")

    def _stop_and_discard(self) -> None:
        self.output("收到停止请求，正在强停并丢弃未完成 Episode")
        self._best_effort_post("/api/force_stop", "强制停止")
        self._best_effort_post("/api/recording/discard", "丢弃未完成录制")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="连续运行自动示教，只保存成功 Episode。"
    )
    parser.add_argument("--target", required=True, choices=TARGET_OBJECTS)
    parser.add_argument("--episodes", required=True, type=int)
    parser.add_argument(
        "--curriculum", default="easy", choices=CURRICULUM_LEVELS
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8767, type=int)
    parser.add_argument("--retry-delay", default=2.0, type=float)
    parser.add_argument(
        "--target-grid",
        metavar="COLUMN,ROW",
        help="将目标物体限制在 full 工作区的指定 5x5 方格",
    )
    parser.add_argument(
        "--max-attempts",
        default=0,
        type=int,
        help="0 表示无限重试（默认）",
    )
    parser.add_argument(
        "--max-failures", default=0, type=int,
        help="累计失败上限；达到后返回 4，交由采集调度器跳过当前组合",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    target_grid = None
    if args.target_grid:
        try:
            target_grid = tuple(
                int(part.strip()) for part in args.target_grid.split(",")
            )
        except ValueError as exc:
            raise SystemExit("--target-grid 必须为 COLUMN,ROW") from exc
        if len(target_grid) != 2:
            raise SystemExit("--target-grid 必须为 COLUMN,ROW")
    runner = ContinuousTrainingRunner(
        api=JsonApiClient(f"http://{args.host}:{args.port}"),
        target=args.target,
        curriculum=args.curriculum,
        episodes=args.episodes,
        target_grid=target_grid,
        retry_delay_s=args.retry_delay,
        max_attempts=args.max_attempts,
        max_failures=args.max_failures,
    )

    def handle_signal(_signum, _frame) -> None:
        runner.request_stop()

    signal.signal(signal.SIGTERM, handle_signal)
    try:
        return runner.run()
    except KeyboardInterrupt:
        runner.request_stop()
        runner._stop_and_discard()
        return 130
    except Exception as exc:
        print(f"连续采集终止：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
