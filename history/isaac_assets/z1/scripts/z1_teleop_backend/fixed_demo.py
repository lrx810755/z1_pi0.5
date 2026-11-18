import threading

import numpy as np

from .config import (
    DEFAULT_SPEED_SCALE,
    FIXED_DEMO_APPROACH_STEPS,
    FIXED_DEMO_APPROACH_TARGETS,
    FIXED_DEMO_ATTACH_MAX_DISTANCE_M,
    FIXED_DEMO_ALLOW_ASSIST,
    FIXED_DEMO_BALL_POSITION,
    FIXED_DEMO_CLOSE_STEPS,
    FIXED_DEMO_LIFT_STEPS,
    FIXED_DEMO_LIFT_TARGETS,
    FIXED_DEMO_MIN_LIFT_DELTA_M,
    FIXED_DEMO_SUPPORT_PATH,
    FIXED_DEMO_USE_FIXED_JOINT,
    GRIPPER_CLOSED,
)


class FixedRedBallDemoController:
    def __init__(self, teleop, ball_manager, attach_body_path):
        self.teleop = teleop
        self.ball_manager = ball_manager
        if isinstance(attach_body_path, (list, tuple)):
            self.attach_body_paths = [path for path in attach_body_path if path]
        else:
            self.attach_body_paths = [attach_body_path] if attach_body_path else []
        self.attach_body_path = self.attach_body_paths[0] if self.attach_body_paths else None
        self.lock = threading.Lock()
        self.active = False
        self.phase = "idle"
        self.phase_step = 0
        self.last_error = ""
        self.last_result = None
        self.pre_lift_ball_z = None
        self._validate_targets(FIXED_DEMO_APPROACH_TARGETS, "approach")
        self._validate_targets(FIXED_DEMO_LIFT_TARGETS, "lift")

    def _validate_targets(self, targets, label):
        missing = []
        out_of_range = {}
        for joint_name, value in targets.items():
            if joint_name not in self.teleop.urdf_limits:
                missing.append(joint_name)
                continue
            lower, upper = self.teleop.safe_bounds(joint_name)
            if not lower <= float(value) <= upper:
                out_of_range[joint_name] = {"value": float(value), "lower": lower, "upper": upper}
        if missing or out_of_range:
            raise RuntimeError(f"固定 demo {label} 路点配置无效: missing={missing} out_of_range={out_of_range}")

    def start(self, speed_scale=DEFAULT_SPEED_SCALE):
        if self.ball_manager is None:
            raise RuntimeError("目标物体管理器尚未初始化")
        with self.lock:
            self.active = True
            self.phase = "approach"
            self.phase_step = 0
            self.last_error = ""
            self.last_result = {
                "ball_position_xyz_m": np.asarray(FIXED_DEMO_BALL_POSITION, dtype=np.float64).tolist(),
                "approach_targets": dict(FIXED_DEMO_APPROACH_TARGETS),
                "lift_targets": dict(FIXED_DEMO_LIFT_TARGETS),
                "assist_enabled": bool(FIXED_DEMO_ALLOW_ASSIST),
                "fixed_joint_enabled": bool(FIXED_DEMO_USE_FIXED_JOINT),
            }
            self.pre_lift_ball_z = None
        self.ball_manager.request_reset(FIXED_DEMO_BALL_POSITION)
        self.teleop.set_named_targets(
            FIXED_DEMO_APPROACH_TARGETS,
                speed_scale=speed_scale,
                message="固定 demo: 目标已放到固定抓取点，正在移动到抓取位",
        )
        return self.snapshot()

    def cancel(self, message="固定 demo 已停止"):
        with self.lock:
            self.active = False
            self.phase = "idle"
            self.phase_step = 0
            self.last_error = message
        self.teleop.hold()
        return self.snapshot()

    def step(self):
        with self.lock:
            if not self.active:
                return
            phase = self.phase
            self.phase_step += 1
            phase_step = self.phase_step

        if phase == "approach" and phase_step >= FIXED_DEMO_APPROACH_STEPS:
            self.teleop.set_named_targets(
                {"jointGripper": GRIPPER_CLOSED},
                speed_scale=max(self.teleop.speed_scale, 0.35),
                message="固定 demo: 到达安全抓取位，正在闭合夹爪",
            )
            with self.lock:
                self.phase = "close"
                self.phase_step = 0
        elif phase == "approach":
            self.ball_manager.hold_world_pose(FIXED_DEMO_BALL_POSITION)
        elif phase == "close" and phase_step >= FIXED_DEMO_CLOSE_STEPS:
            attachment = None
            self.ball_manager.hold_world_pose(FIXED_DEMO_BALL_POSITION)
            snapshot = self.ball_manager.snapshot()
            position = snapshot.get("position_xyz_m")
            self.pre_lift_ball_z = None if position is None else float(position[2])
            if FIXED_DEMO_ALLOW_ASSIST:
                try:
                    attachment = self.ball_manager.attach_to_body_assisted(
                        self.attach_body_path,
                        expected_ball_position=FIXED_DEMO_BALL_POSITION,
                        max_distance=FIXED_DEMO_ATTACH_MAX_DISTANCE_M,
                    )
                except Exception as exc:
                    with self.lock:
                        self.active = False
                        self.phase = "idle"
                        self.phase_step = 0
                        self.last_error = f"固定 demo 辅助附着失败: {exc}"
                    self.teleop.hold()
                    return
            elif FIXED_DEMO_USE_FIXED_JOINT:
                try:
                    attachment = self.ball_manager.attach_to_physics_joint(
                        self.attach_body_paths,
                        expected_ball_position=FIXED_DEMO_BALL_POSITION,
                        force=True,
                    )
                except Exception as exc:
                    with self.lock:
                        self.active = False
                        self.phase = "idle"
                        self.phase_step = 0
                        self.last_error = f"固定 demo 固定关节抓取失败: {exc}"
                    self.teleop.hold()
                    return
            if attachment is not None:
                support_prim = self.ball_manager.world.stage.GetPrimAtPath(FIXED_DEMO_SUPPORT_PATH)
                if support_prim.IsValid():
                    self.ball_manager.world.stage.RemovePrim(FIXED_DEMO_SUPPORT_PATH)
            self.teleop.set_named_targets(
                FIXED_DEMO_LIFT_TARGETS,
                speed_scale=max(self.teleop.speed_scale, 0.35),
                message="固定 demo: 夹爪已闭合，正在物理抬升验证",
            )
            with self.lock:
                self.phase = "lift"
                self.phase_step = 0
                if self.last_result is None:
                    self.last_result = {}
                self.last_result["pre_lift_ball_z_m"] = self.pre_lift_ball_z
                if attachment is not None:
                    self.last_result["attachment"] = {
                        "attached": attachment.get("attached", False),
                        "attached_body_path": attachment.get("attached_body_path"),
                        "attachment_mode": attachment.get("attachment_mode"),
                        "attach_distance_m": attachment.get("last_attach_distance_m"),
                    }
        elif phase == "close":
            self.ball_manager.hold_world_pose(FIXED_DEMO_BALL_POSITION)
        elif phase == "lift" and phase_step >= FIXED_DEMO_LIFT_STEPS:
            snapshot = self.ball_manager.snapshot()
            position = snapshot.get("position_xyz_m")
            ball_z = None if position is None else float(position[2])
            lift_delta = None if ball_z is None or self.pre_lift_ball_z is None else ball_z - self.pre_lift_ball_z
            if (
                not FIXED_DEMO_ALLOW_ASSIST
                and (lift_delta is None or lift_delta < FIXED_DEMO_MIN_LIFT_DELTA_M)
            ):
                with self.lock:
                    self.active = False
                    self.phase = "idle"
                    self.phase_step = 0
                    self.last_error = (
                        f"固定 demo 物理抓取失败: 目标上升 {0.0 if lift_delta is None else lift_delta:.3f}m，"
                        f"需要 >= {FIXED_DEMO_MIN_LIFT_DELTA_M:.3f}m；未使用吸附辅助"
                    )
                    if self.last_result is None:
                        self.last_result = {}
                    self.last_result["post_lift_ball_z_m"] = ball_z
                    self.last_result["lift_delta_m"] = lift_delta
                self.teleop.hold()
                return
            with self.lock:
                self.active = False
                self.phase = "done"
                self.phase_step = 0
                if self.last_result is None:
                    self.last_result = {}
                self.last_result["post_lift_ball_z_m"] = ball_z
                self.last_result["lift_delta_m"] = lift_delta
            self.teleop.set_named_targets(
                {"jointGripper": GRIPPER_CLOSED},
                speed_scale=self.teleop.speed_scale,
                message="固定 demo 完成: 目标已抓取并抬升",
            )

    def snapshot(self):
        with self.lock:
            return {
                "active": self.active,
                "phase": self.phase,
                "phase_step": self.phase_step,
                "error": self.last_error,
                "attach_body_path": self.attach_body_path,
                "attach_body_paths": list(self.attach_body_paths),
                "last_result": dict(self.last_result) if self.last_result else None,
            }
