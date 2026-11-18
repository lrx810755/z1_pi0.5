"""Pure safety and reporting helpers for SmolVLA simulation rollout."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


JOINT_NAMES = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    "jointGripper",
)
JOINT_LIMITS = np.asarray(
    [
        (-math.radians(150.0), math.radians(150.0)),
        (0.0, math.radians(170.0)),
        (-math.radians(165.0), 0.0),
        (-math.radians(87.0), math.radians(87.0)),
        (-math.radians(77.0), math.radians(77.0)),
        (-math.radians(160.0), math.radians(160.0)),
        (-math.radians(90.0), 0.0),
    ],
    dtype=np.float64,
)
TASK_INSTRUCTIONS = {
    "cup": "Physically grasp the cup, lift it clear, and place it in the box.",
    "apple": "Physically grasp the apple, lift it clear, and place it in the box.",
    "peach": "Physically grasp the peach, lift it clear, and place it in the box.",
    "beverage_bottle": (
        "Physically grasp the bottle, lift it clear, and place it in the box."
    ),
}
CAMERA_KEYS = ("wrist", "vla_wrist", "global")


def camera_keys_for_features(features: Iterable[str]) -> tuple[str, ...]:
    configured = set(features)
    cameras = tuple(
        name
        for name in (*CAMERA_KEYS, "global_right")
        if f"observation.images.{name}" in configured
    )
    if not cameras:
        raise ValueError("checkpoint must provide at least one camera")
    return cameras


class GraspPhaseTracker:
    """Translate policy gripper motion into evaluator control phases."""

    def __init__(self) -> None:
        self.previous_command: float | None = None
        self.was_grasped = False
        self.close_peak: float | None = None
        self.lift_started = False
        self.release_started = False
        self.release_complete = False

    def update(
        self,
        command_gripper: float,
        actual_gripper: float,
        status: Mapping[str, object],
        *,
        gripper_target: float | None = None,
    ) -> dict[str, object]:
        # Binary intent stays closed even when contact changes the limited
        # motor command. Only a new open target should begin release.
        command = float(command_gripper if gripper_target is None else gripper_target)
        actual = float(actual_gripper)
        if not math.isfinite(command) or not math.isfinite(actual):
            raise ValueError("gripper phase inputs must be finite")
        self.was_grasped = self.was_grasped or bool(
            status.get("physical_grasp_ok", False)
        )
        # Detect accumulated opening, independent of command frequency.
        if self.was_grasped:
            self.close_peak = max(command, self.close_peak if self.close_peak is not None else command)
        opening = self.close_peak is not None and command < self.close_peak - 0.005
        if self.was_grasped and opening:
            self.release_started = True
        if self.release_started and actual <= -0.80:
            self.release_complete = True
        commanded_closed = command >= -0.75 and not self.release_started
        if commanded_closed and bool(status.get("bilateral_contact", False)):
            self.lift_started = True
        self.previous_command = command
        return {
            "gripper_commanded_closed": commanded_closed,
            "gripper_obstructed": bool(
                commanded_closed and status.get("bilateral_contact", False)
            ),
            "physical_test_lift_active": bool(
                self.lift_started and not self.release_started
            ),
            "physical_grasp_ok": False,
            "release_started": self.release_started,
            "release_sequence_complete": self.release_complete,
            "transport_failure_reason": "",
        }


class PreliftContactGate:
    """Hold the arm during closing until reported bilateral contact settles."""

    def __init__(self, stable_s: float, timeout_s: float, max_gap_s: float = 0.5) -> None:
        self.stable_s = stable_s
        self.max_gap_s = max_gap_s
        self.last_observation = None
        self.timeout_s = timeout_s
        self.started = None
        self.contact_since = None
        self.hold_arm = None
        self.ready = False

    def apply(self, command, current, now):
        command = np.asarray(command, dtype=float).copy()
        if self.stable_s <= 0:
            return command
        if command[-1] <= -0.80:
            self.started = self.contact_since = self.hold_arm = None
            self.ready = False
            return command
        if self.started is None and command[-1] >= -0.75:
            self.started = now
            self.hold_arm = np.asarray(current[:6], dtype=float).copy()
        if self.started is None or self.ready:
            return command
        ready_at = self.contact_since + self.stable_s if self.contact_since is not None else None
        if now - self.started >= self.timeout_s and (
            ready_at is None or ready_at > self.started + self.timeout_s + 1e-9
        ):
            raise RuntimeError("prelift_contact_not_stable")
        self.ready = ready_at is not None and now >= ready_at - 1e-9
        if not self.ready:
            command[:6] = self.hold_arm
        return command

    def observe(self, status, now):
        """Called from telemetry callbacks, including during model inference."""
        if self.started is None or self.ready:
            return
        if self.last_observation is not None and now - self.last_observation > self.max_gap_s + 1e-9:
            self.contact_since = None
        self.last_observation = now
        if status.get("bilateral_contact", False):
            if self.contact_since is None:
                self.contact_since = now
        else:
            self.contact_since = None


class GripperCloseHold:
    """Keep a newly issued binary close target for a minimum simulation time."""
    def __init__(self, duration_s: float = 0.0):
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("close hold duration must be finite and nonnegative")
        self.duration_s = duration_s
        self.closed = False
        self.until = None
        self.blocked_open = False

    def apply(self, target: float, sim_time: float) -> float:
        self.blocked_open = False
        if target == 0.0:
            if not self.closed:
                self.until = sim_time + self.duration_s
                self.closed = True
            return target
        if self.closed and sim_time < self.until:
            self.blocked_open = True
            return 0.0
        self.closed = False
        self.until = None
        return target


def binary_gripper_target(requested: Sequence[float], threshold_rad: float = -0.75,
                          encoding: str = "angle", *,
                          previous_target: float | None = None) -> np.ndarray:
    """Map a Z1 policy gripper angle to its demonstrated open/closed targets."""
    target = np.asarray(requested, dtype=np.float64).copy()
    if target.shape != (len(JOINT_NAMES),) or not np.isfinite(target).all():
        raise ValueError("expected 7 finite policy actions")
    if not math.isfinite(threshold_rad) or not -0.90 < threshold_rad < 0.0:
        raise ValueError("gripper threshold must be between -0.90 and 0.0 rad")
    if encoding not in {"angle", "binary"}:
        raise ValueError("unknown gripper action encoding")
    if encoding == "binary" and previous_target is not None:
        if previous_target not in (-0.90, 0.0):
            raise ValueError("previous gripper target must be -0.90 or 0.0")
        value = target[-1]
        target[-1] = 0.0 if value >= 0.6 else (-0.90 if value <= 0.4 else previous_target)
        return target
    threshold = 0.5 if encoding == "binary" else threshold_rad
    target[-1] = 0.0 if target[-1] >= threshold else -0.90
    return target


def safe_joint_target(
    requested: Sequence[float],
    current: Sequence[float],
    max_step_rad: float = 0.10,
) -> np.ndarray:
    requested_array = np.asarray(requested, dtype=np.float64).reshape(-1)
    current_array = np.asarray(current, dtype=np.float64).reshape(-1)
    if requested_array.shape != (len(JOINT_NAMES),):
        raise ValueError(f"expected 7 policy actions, got {requested_array.shape}")
    if current_array.shape != requested_array.shape:
        raise ValueError(f"expected 7 joint states, got {current_array.shape}")
    if not np.all(np.isfinite(requested_array)) or not np.all(np.isfinite(current_array)):
        raise ValueError("policy action and joint state must be finite")
    if not math.isfinite(max_step_rad) or max_step_rad <= 0.0:
        raise ValueError("max_step_rad must be finite and positive")
    within_limits = np.clip(requested_array, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    stepped = current_array + np.clip(
        within_limits - current_array, -max_step_rad, max_step_rad
    )
    return np.clip(stepped, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def execution_window(action_chunk, steps: int = 10) -> np.ndarray:
    chunk = np.asarray(action_chunk)
    if chunk.ndim == 3:
        if chunk.shape[0] != 1:
            raise ValueError("rollout only supports batch size 1")
        chunk = chunk[0]
    if chunk.ndim != 2 or chunk.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"invalid action chunk shape {chunk.shape}")
    if steps <= 0:
        raise ValueError("execution steps must be positive")
    return chunk[:steps].copy()


def pregrasp_collision_reason(
    status: Mapping[str, object],
    target: str,
    gripper_command: float,
    target_position: Sequence[float] | None,
) -> str | None:
    if gripper_command >= -0.75 or status.get("target_object") != target:
        return None
    if target in (
        status.get("left_finger_contact_object"),
        status.get("right_finger_contact_object"),
    ):
        return "pregrasp_target_contact"
    if target_position is None:
        return None
    try:
        current = np.asarray(status["object_position_m"], dtype=float)
        initial = np.asarray(target_position, dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        current.size >= 2 and initial.size >= 2
        and np.isfinite(current[:2]).all()
        and np.isfinite(initial[:2]).all()
        and np.linalg.norm(current[:2] - initial[:2]) > 0.015
    ):
        return "pregrasp_target_moved"
    return None


def streams_are_fresh(
    stamps: Mapping[str, float],
    now: float,
    timeout_s: float = 0.5,
    camera_keys: Sequence[str] = CAMERA_KEYS,
) -> bool:
    required = {"joint", *camera_keys}
    return required.issubset(stamps) and all(
        math.isfinite(stamps[name]) and 0.0 <= now - stamps[name] <= timeout_s
        for name in required
    )


def target_in_xy_bounds(
    scene_status: Mapping[str, object],
    target: str,
    bounds: Sequence[float],
) -> bool:
    """Accept only settled target poses inside a measured training rectangle."""
    try:
        position = scene_status["object_positions_m"][target]
        x, y = float(position[0]), float(position[1])
        return (
            math.isfinite(x) and math.isfinite(y)
            and bounds[0] <= x <= bounds[1]
            and bounds[2] <= y <= bounds[3]
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def arm_tracking_error(command: Sequence[float], current: Sequence[float]) -> float:
    command_array = np.asarray(command, dtype=np.float64).reshape(-1)
    current_array = np.asarray(current, dtype=np.float64).reshape(-1)
    if command_array.shape != (len(JOINT_NAMES),) or current_array.shape != command_array.shape:
        raise ValueError("tracking error requires seven joint values")
    if not np.all(np.isfinite(command_array)) or not np.all(np.isfinite(current_array)):
        raise ValueError("tracking error inputs must be finite")
    return float(np.max(np.abs(command_array[:6] - current_array[:6])))


class BottleTipGuard:
    """Latch a fall after >70 degrees for one continuous simulation second."""

    def __init__(self, start_stamp: float = -1.0):
        self.start_stamp = start_stamp
        self.last_stamp = None
        self.tipped_since = None
        self.placement_started = False
        self.tripped = False

    def observe(self, orientation, stamp: float, placing: bool) -> bool:
        if self.tripped:
            return True
        if not math.isfinite(stamp) or stamp < self.start_stamp:
            return False
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return False
        if self.last_stamp is not None and stamp - self.last_stamp > 0.25:
            # Missing truth samples cannot prove continuous tilt.
            self.tipped_since = None
        self.last_stamp = stamp
        self.placement_started = self.placement_started or placing
        if self.placement_started:
            self.tipped_since = None
            return False
        q = np.asarray(orientation, dtype=float)
        norm = float(np.linalg.norm(q))
        if q.shape != (4,) or not np.all(np.isfinite(q)) or norm < 1e-12:
            self.tipped_since = None
            return False
        x, y, z, w = q / norm
        # Local +Z is the bottle long axis; quaternion order is xyzw.
        up_z = 1.0 - 2.0 * (x*x + y*y)
        tipped = up_z < math.cos(math.radians(70.0)) - 1e-12
        if not tipped:
            self.tipped_since = None
        elif self.tipped_since is None:
            self.tipped_since = stamp
        elif stamp - self.tipped_since >= 1.0 - 1e-9:
            self.tripped = True
        return self.tripped


def classify_failure(
    status: Mapping[str, object],
    timed_out: bool,
    stop_reason: str = "",
) -> str:
    if bool(status.get("task_success", False)):
        return "success"
    if stop_reason:
        return stop_reason
    if not timed_out:
        return "stopped"
    if not bool(status.get("physical_grasp_ok", False)) or not bool(
        status.get("lifted", False)
    ):
        return "grasp_failed"
    if not bool(status.get("release_sequence_complete", False)):
        return "transport_or_release_failed"
    return "placement_failed"


def build_report(
    results: Iterable[Mapping[str, object]],
    expected_objects: Sequence[str] | None = None,
    expected_per_object: int | None = None,
) -> dict:
    rows = list(results)
    grouped = defaultdict(list)
    failures = Counter()
    for row in rows:
        grouped[str(row["target_object"])].append(row)
        category = str(row.get("failure_category", "unknown"))
        if category != "success":
            failures[category] += 1

    def summary(items):
        successes = sum(bool(item.get("success", False)) for item in items)
        count = len(items)
        return {
            "episodes": count,
            "successes": successes,
            "success_rate": successes / count if count else 0.0,
        }

    per_object = {
        name: summary(items) for name, items in sorted(grouped.items())
    }
    overall = summary(rows)
    object_rates_pass = bool(per_object) and all(
        item["success_rate"] >= 0.70 for item in per_object.values()
    )
    complete = True
    if expected_objects is not None:
        complete = set(per_object) == set(expected_objects)
    if expected_per_object is not None:
        complete = complete and all(
            item["episodes"] >= expected_per_object
            for item in per_object.values()
        )
    return {
        "overall": overall,
        "per_object": per_object,
        "failure_categories": dict(sorted(failures.items())),
        "acceptance": {
            "per_object_minimum": 0.70,
            "overall_minimum": 0.75,
            "complete": complete,
            "passed": (
                complete and object_rates_pass
                and overall["success_rate"] >= 0.75
            ),
        },
    }


def load_training_seeds(manifest_path: str | Path) -> set[int]:
    manifest_path = Path(manifest_path).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest["source"]).expanduser()
    seeds = set()
    for episode_names in manifest.get("objects", {}).values():
        for episode_name in episode_names:
            summary_path = source / episode_name / "episode_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            seed = summary.get("scene_randomization", {}).get("seed")
            if seed is not None:
                seeds.add(int(seed))
    return seeds
