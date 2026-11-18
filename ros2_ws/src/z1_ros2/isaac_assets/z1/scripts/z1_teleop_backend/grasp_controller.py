import threading

import numpy as np
from isaacsim.core.utils.xforms import get_world_pose

from .camera import (
    camera_point_from_pixel_depth,
    depth_from_camera,
    transform_camera_point_to_world,
)
from .config import (
    DEFAULT_SPEED_SCALE,
    GRASP_APPROACH_OFFSET_M,
    GRASP_CLOSE_SETTLE_STEPS,
    GRASP_LIFT_DISTANCE_M,
    GRASP_LIFT_VERIFY_MIN_DELTA_Z_M,
    GRASP_MIN_GRASP_Z_M,
    GRASP_MIN_PREGRASP_Z_M,
    GRASP_PHYSICS_FALLBACK_SETTLE_STEPS,
    GRASP_PREGRASP_DISTANCE_M,
    GRASP_USE_PHYSICS_TARGET_FALLBACK,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    RED_BALL_MIN_COMPLETE_FRAMES,
)
from .vision_red_ball import detect_red_ball


class RedBallGraspController:
    def __init__(
        self,
        teleop,
        camera,
        calibration,
        mount_path,
        gripper_path,
        gripper_attach_paths,
        ball_manager,
        moveit_bridge,
        trajectory_executor,
    ):
        self.teleop = teleop
        self.camera = camera
        self.calibration = calibration
        self.mount_path = mount_path
        self.gripper_path = gripper_path
        self.gripper_attach_paths = list(gripper_attach_paths)
        self.ball_manager = ball_manager
        self.moveit = moveit_bridge
        self.trajectory_executor = trajectory_executor
        self.lock = threading.Lock()
        self.active = False
        self.phase = "idle"
        self.phase_step = 0
        self.last_result = None
        self.last_error = ""
        self.target_world = None
        self.grasp_plan = None
        self.grasp_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.moveit_started = False
        self.last_detection = None
        self.complete_detection_frames = 0
        self.pre_lift_ball_z = None
        self.last_attachment = None

    def start(self, speed_scale=DEFAULT_SPEED_SCALE):
        with self.lock:
            self.active = True
            self.phase = "detect"
            self.phase_step = 0
            self.last_error = ""
            self.last_result = None
            self.target_world = None
            self.grasp_plan = None
            self.moveit_started = False
            self.last_detection = None
            self.complete_detection_frames = 0
            self.pre_lift_ball_z = None
            self.last_attachment = None
        self.moveit.cancel("")
        self.trajectory_executor.cancel("")
        self.teleop.hold()
        self.teleop.set_speed(speed_scale)
        with self.teleop.lock:
            self.teleop.last_message = "抓取识别已启动，请手动操作至完整红球进入摄像头画面"
        return self.snapshot()

    def cancel(self, message="红球抓取任务已停止"):
        with self.lock:
            self.active = False
            self.phase = "idle"
            self.phase_step = 0
            self.last_error = message
            self.moveit_started = False
        self.moveit.cancel()
        self.trajectory_executor.cancel(message)
        self.teleop.set_external_arm_control(False, message)
        return self.snapshot()

    def snapshot(self):
        with self.lock:
            result = dict(self.last_result) if self.last_result else None
            plan = dict(self.grasp_plan) if self.grasp_plan else None
            detection = self.detection_snapshot_locked()
            return {
                "active": self.active,
                "phase": self.phase,
                "phase_step": self.phase_step,
                "error": self.last_error,
                "last_detection": detection,
                "complete_detection_frames": self.complete_detection_frames,
                "required_complete_detection_frames": RED_BALL_MIN_COMPLETE_FRAMES,
                "last_attachment": self.last_attachment,
                "moveit_state": self.moveit.snapshot(),
                "trajectory_state": self.trajectory_executor.snapshot(),
                "target_world_xyz_m": self.target_world.tolist() if self.target_world is not None else None,
                "last_result": result,
                "plan": plan,
            }

    def detection_snapshot_locked(self):
        if not self.last_detection:
            return None
        snapshot = {}
        for key, value in self.last_detection.items():
            if isinstance(value, np.ndarray):
                snapshot[key] = value.astype(float).tolist()
            elif isinstance(value, np.generic):
                snapshot[key] = value.item()
            else:
                snapshot[key] = value
        return snapshot

    def step(self):
        with self.lock:
            if not self.active:
                return
            phase = self.phase
            self.phase_step += 1

        if phase == "detect":
            self._detect_and_plan()
        elif phase == "plan_pregrasp":
            self._start_moveit_phase("execute_pregrasp", "pregrasp")
        elif phase == "planning_pregrasp":
            self._finish_moveit_phase("execute_pregrasp", "pregrasp")
        elif phase == "execute_pregrasp":
            self._finish_trajectory_phase("pregrasp", "plan_grasp")
        elif phase == "plan_grasp":
            self._start_moveit_phase("execute_grasp", "grasp")
        elif phase == "planning_grasp":
            self._finish_moveit_phase("execute_grasp", "grasp")
        elif phase == "execute_grasp":
            self._finish_trajectory_phase("grasp", "close_gripper")
        elif phase == "close_gripper":
            self._close_phase()
        elif phase == "plan_lift":
            self._start_moveit_phase("execute_lift", "lift")
        elif phase == "planning_lift":
            self._finish_moveit_phase("execute_lift", "lift")
        elif phase == "execute_lift":
            self._finish_trajectory_phase("lift", "done")
        elif phase == "done":
            self._finish_done()
        elif phase == "close":
            self._close_phase()

    def _detect_and_plan(self):
        detection = detect_red_ball(self.camera)
        if not detection.get("found", False):
            if GRASP_USE_PHYSICS_TARGET_FALLBACK and self._try_physics_target_fallback("camera_detection_missing"):
                return
            with self.lock:
                self.last_detection = detection
                self.complete_detection_frames = 0
                self.last_error = (
                    f"等待红色像素: red_pixels={detection.get('red_pixels', 0)} "
                    f"reason={detection.get('reason', 'unknown')}"
                )
            return

        if not detection.get("complete", False):
            if GRASP_USE_PHYSICS_TARGET_FALLBACK and self._try_physics_target_fallback("camera_detection_incomplete"):
                return
            with self.lock:
                self.last_detection = detection
                self.complete_detection_frames = 0
                self.last_error = (
                    f"红球未完整入画: bbox={detection.get('bbox')} "
                    f"red_pixels={detection.get('red_pixels', 0)} "
                    f"radius={float(detection.get('radius_px', 0.0)):.1f}px "
                    f"circularity={detection.get('circularity')}"
                )
            return

        with self.lock:
            self.last_detection = detection
            self.complete_detection_frames += 1
            if self.complete_detection_frames < RED_BALL_MIN_COMPLETE_FRAMES:
                self.last_error = (
                    f"完整红球确认中: {self.complete_detection_frames}/"
                    f"{RED_BALL_MIN_COMPLETE_FRAMES}"
                )
                return

        depth_m, depth_source = depth_from_camera(self.camera, detection, self.calibration)
        camera_point = camera_point_from_pixel_depth(detection["pixel"], depth_m, self.calibration)
        mount_point, world_point = transform_camera_point_to_world(
            camera_point,
            self.calibration,
            self.mount_path,
        )
        self._plan_from_world_target(
            world_point,
            target_source="camera_detection",
            detection=detection,
            camera_point=camera_point,
            mount_point=mount_point,
            depth_m=depth_m,
            depth_source=depth_source,
        )

    def _try_physics_target_fallback(self, reason):
        if not GRASP_USE_PHYSICS_TARGET_FALLBACK or self.ball_manager is None:
            return False
        with self.lock:
            if self.phase_step < GRASP_PHYSICS_FALLBACK_SETTLE_STEPS:
                return False
        snapshot = self.ball_manager.snapshot()
        if not snapshot.get("prim_valid") or snapshot.get("position_xyz_m") is None:
            return False
        world_point = np.asarray(snapshot["position_xyz_m"], dtype=np.float64)
        self._plan_from_world_target(world_point, target_source=f"physics_target_fallback:{reason}")
        return True

    def _plan_from_world_target(
        self,
        world_point,
        target_source,
        detection=None,
        camera_point=None,
        mount_point=None,
        depth_m=None,
        depth_source=None,
    ):
        world_point = np.asarray(world_point, dtype=np.float64)
        gripper_position, gripper_quat = get_world_pose(self.gripper_path)
        gripper_position = np.asarray(gripper_position, dtype=np.float64)
        approach_dir = gripper_position - world_point
        if np.linalg.norm(approach_dir) < 1e-6:
            approach_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        pregrasp = world_point + approach_dir * GRASP_PREGRASP_DISTANCE_M
        grasp = world_point + approach_dir * GRASP_APPROACH_OFFSET_M
        raw_grasp = grasp.copy()
        grasp[2] = max(float(grasp[2]), GRASP_MIN_GRASP_Z_M)
        pregrasp[2] = max(float(pregrasp[2]), GRASP_MIN_PREGRASP_Z_M, float(grasp[2]) + 0.04)
        lift = grasp + np.array([0.0, 0.0, GRASP_LIFT_DISTANCE_M], dtype=np.float64)
        lift[2] = max(float(lift[2]), GRASP_MIN_PREGRASP_Z_M)
        grasp_orientation = np.asarray(gripper_quat, dtype=np.float64)

        last_result = {
            "target_source": target_source,
            "world_xyz_m": world_point.tolist(),
        }
        if self.ball_manager is not None:
            ball_snapshot = self.ball_manager.snapshot()
            ball_position = ball_snapshot.get("position_xyz_m")
            if ball_position is not None:
                ball_position = np.asarray(ball_position, dtype=np.float64)
                error = world_point - ball_position
                last_result["red_ball_entity_xyz_m"] = ball_position.tolist()
                last_result["vision_entity_error_xyz_m"] = error.tolist()
                last_result["vision_entity_error_norm_m"] = float(np.linalg.norm(error))
        if detection is not None:
            last_result.update(
                {
                    "pixel_uv": [float(detection["pixel"][0]), float(detection["pixel"][1])],
                    "area_px": detection["area_px"],
                    "bbox": detection["bbox"],
                    "radius_px": detection["radius_px"],
                    "circularity": detection.get("circularity"),
                    "complete": detection.get("complete", False),
                    "depth_m": float(depth_m),
                    "depth_source": depth_source,
                    "camera_xyz_m": camera_point[:3].tolist(),
                    "mount_xyz_m": mount_point.tolist(),
                    "math": {
                        "camera_matrix_K": self.calibration["camera_matrix_K"].tolist(),
                        "T_mount_camera": self.calibration["T_mount_camera"].tolist(),
                        "formula": "P_camera=[(u-cx)/fx*z, -(v-cy)/fy*z, -z, 1]; P_mount=T_mount_camera*P_camera; P_world=T_world_mount*P_mount",
                    },
                }
            )

        with self.lock:
            self.target_world = world_point
            self.grasp_orientation_wxyz = np.asarray(grasp_orientation, dtype=np.float64)
            self.grasp_plan = {
                "pregrasp_xyz_m": pregrasp.tolist(),
                "grasp_xyz_m": grasp.tolist(),
                "raw_grasp_xyz_m": raw_grasp.tolist(),
                "lift_xyz_m": lift.tolist(),
                "approach_dir": approach_dir.tolist(),
                "link06_to_gripper_contact_offset_m": GRASP_APPROACH_OFFSET_M,
                "min_grasp_z_m": GRASP_MIN_GRASP_Z_M,
                "orientation_wxyz": self.grasp_orientation_wxyz.tolist(),
                "orientation_source": "current_gripper_pose",
            }
            self.last_result = last_result
            self.phase = "plan_pregrasp"
            self.phase_step = 0
            self.last_error = ""
            self.moveit_started = False
            self.complete_detection_frames = 0
            self.pre_lift_ball_z = None
            self.last_attachment = None

        self.teleop.set_named_targets(
            {"jointGripper": GRIPPER_OPEN},
            message=f"抓取目标已确定({target_source}): world={np.round(world_point, 3).tolist()}",
        )

    def _start_moveit_phase(self, next_phase, plan_key):
        with self.lock:
            if self.grasp_plan is None:
                self._fail_locked("没有可执行的抓取规划点")
                return
            if self.moveit_started:
                return
            position = np.asarray(self.grasp_plan[f"{plan_key}_xyz_m"], dtype=np.float64)
            orientation = self.grasp_orientation_wxyz.copy()
            speed_scale = self.teleop.speed_scale
            start_joint_state = self.teleop.moveit_start_state()
            self.moveit_started = True

        started = self.moveit.plan_async(
            plan_key,
            position,
            orientation,
            speed_scale=speed_scale,
            start_joint_state=start_joint_state,
        )
        with self.lock:
            if not started:
                self.moveit_started = False
                return
            self.phase = f"planning_{plan_key}"
            self.phase_step = 0
            self.last_error = ""
        self.teleop.set_named_targets({}, message=f"MoveIt 正在规划 {plan_key} 轨迹")

    def _finish_moveit_phase(self, execute_phase, plan_key):
        state = self.moveit.snapshot()
        if state["active"]:
            return
        if state["error"]:
            with self.lock:
                self._fail_locked(state["error"])
            return
        if state["result"] is None:
            return
        trajectory = (state["result"] or {}).get("trajectory") or {}
        try:
            self.trajectory_executor.start(plan_key, trajectory)
        except Exception as exc:
            with self.lock:
                self._fail_locked(str(exc))
            return
        with self.lock:
            self.phase = execute_phase
            self.phase_step = 0
            self.moveit_started = False
            self.last_error = ""

    def _finish_trajectory_phase(self, trajectory_phase, next_phase):
        trajectory_state = self.trajectory_executor.snapshot()
        if trajectory_state["error"]:
            with self.lock:
                self._fail_locked(trajectory_state["error"])
            return
        if trajectory_state["active"]:
            return
        if not self.trajectory_executor.consume_completed(trajectory_phase):
            return
        with self.lock:
            self.phase = next_phase
            self.phase_step = 0
            self.last_error = ""

    def _close_phase(self):
        with self.lock:
            if self.phase_step == 1:
                self.teleop.set_named_targets(
                    {"jointGripper": GRIPPER_CLOSED},
                    speed_scale=max(self.teleop.speed_scale, 0.35),
                    message="到达抓取位，正在闭合夹爪进行物理抓取",
                )
            if self.phase_step < GRASP_CLOSE_SETTLE_STEPS:
                return
            if self.ball_manager is not None:
                try:
                    snapshot = self.ball_manager.snapshot()
                    position = snapshot.get("position_xyz_m")
                    self.pre_lift_ball_z = None if position is None else float(position[2])
                    expected_ball_position = None if self.target_world is None else self.target_world.copy()
                    attachment = self.ball_manager.attach_to_physics_joint(
                        self.gripper_attach_paths,
                        expected_ball_position=expected_ball_position,
                    )
                    self.last_attachment = {
                        "attached": attachment.get("attached", False),
                        "attached_body_path": attachment.get("attached_body_path"),
                        "attachment_mode": attachment.get("attachment_mode"),
                        "attach_distance_m": attachment.get("last_attach_distance_m"),
                        "body_origin_distance_m": attachment.get("last_attach_body_origin_distance_m"),
                        "expected_ball_position_xyz_m": (
                            None if expected_ball_position is None else np.asarray(expected_ball_position).tolist()
                        ),
                    }
                except Exception as exc:
                    self._fail_locked(f"物理抓取约束创建失败: {exc}")
                    return
            self.phase = "plan_lift"
            self.phase_step = 0
            self.moveit_started = False

    def _finish_done(self):
        if self.ball_manager is not None and self.pre_lift_ball_z is not None:
            snapshot = self.ball_manager.snapshot()
            position = snapshot.get("position_xyz_m")
            if position is None:
                with self.lock:
                    self._fail_locked("抬升验证失败: 红球位置无效")
                return
            lift_delta = float(position[2]) - float(self.pre_lift_ball_z)
            if lift_delta < GRASP_LIFT_VERIFY_MIN_DELTA_Z_M:
                with self.lock:
                    self._fail_locked(
                        f"抬升验证失败: 红球仅上升 {lift_delta:.3f}m，"
                        f"需要 >= {GRASP_LIFT_VERIFY_MIN_DELTA_Z_M:.3f}m"
                    )
                return
        with self.lock:
            self.phase = "done"
            self.active = False
        self.teleop.set_external_arm_control(False, "MoveIt 抓取完成，后端恢复关节控制")
        self.teleop.set_named_targets(
            {"jointGripper": GRIPPER_CLOSED},
            speed_scale=self.teleop.speed_scale,
            message="红球物理抓取流程完成",
        )

    def _fail_locked(self, message):
        self.active = False
        self.phase = "idle"
        self.phase_step = 0
        self.last_error = message
        self.moveit_started = False
        self.teleop.set_external_arm_control(False, message)
