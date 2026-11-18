"""Synchronize dynamic task objects with MoveIt."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from typing import Any, Dict, Tuple

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from .grasp_evaluator import (
    OBJECT_FRAME_NAMES,
    canonical_object_name,
    quaternion_angle,
    rotate_vector,
)
from .grasp_config import GRASP_CONFIG


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
PoseSample = Tuple[float, Vector3, Quaternion]

APPLE_PROXY_RADII = tuple(
    float(value)
    for value in GRASP_CONFIG["collision_proxy"]["apple"]["radii_m"]
)

OBJECT_COLLISIONS: Dict[str, Dict[str, Any]] = {
    "cup": {"shape": "cylinder", "radius": 0.0252, "height": 0.061},
    # Conservative planning sphere for the 35.2 mm-wide apple-body proxy.
    # The narrow visual stem is intentionally excluded from collision.
    "apple": {
        "shape": "sphere",
        "radius": max(APPLE_PROXY_RADII[:2]) + 0.0003,
    },
    "peach": {"shape": "sphere", "radius": 0.026},
    "mouse": {"shape": "box", "size": (0.084, 0.048, 0.028)},
    "beverage_bottle": {
        "shape": "cylinder",
        "radius": 0.0242,
        "height": 0.157,
    },
}


class ObjectCollisionSync(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_object_collision_sync")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter(
            "touch_links",
            ["link06", "camera_link", "gripperStator", "gripperMover"],
        )
        self.declare_parameter("update_rate_hz", 5.0)
        self.declare_parameter("position_update_threshold_m", 0.002)
        self.declare_parameter("rotation_update_threshold_deg", 2.0)
        self.declare_parameter("object_timeout_s", 1.0)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.touch_links = [
            str(value) for value in self.get_parameter("touch_links").value
        ]
        update_rate = float(self.get_parameter("update_rate_hz").value)
        self.position_threshold = float(
            self.get_parameter("position_update_threshold_m").value
        )
        self.rotation_threshold = math.radians(
            float(self.get_parameter("rotation_update_threshold_deg").value)
        )
        self.object_timeout = float(
            self.get_parameter("object_timeout_s").value
        )

        self._lock = threading.RLock()
        self._poses: Dict[str, PoseSample] = {}
        self._last_applied_poses: Dict[str, PoseSample] = {}
        self._target_object = "apple"
        self._grasp_success = False
        self._last_applied_target = ""
        self._base_acm = None
        self._get_pending = False
        self._apply_pending = False
        self._force_update = True
        self._attempts = 0
        self._last_error = ""

        callback_group = ReentrantCallbackGroup()
        self._apply_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
            callback_group=callback_group,
        )
        self._get_client = self.create_client(
            GetPlanningScene,
            "/get_planning_scene",
            callback_group=callback_group,
        )
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._status_pub = self.create_publisher(
            String, "/smolvla/object_collision_status", latched_qos
        )
        self.create_subscription(
            TFMessage,
            "/isaac/object_transforms",
            self._object_callback,
            20,
            callback_group=callback_group,
        )
        self.create_subscription(
            String,
            "/smolvla/status",
            self._grasp_status_callback,
            10,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "/smolvla/reload_object_collisions",
            self._reload_callback,
            callback_group=callback_group,
        )
        self._publish_status("WAITING", "等待 MoveIt ACM 与物体位姿")
        self._timer = self.create_timer(
            1.0 / max(update_rate, 1.0),
            self._update,
            callback_group=callback_group,
        )

    @staticmethod
    def _object_id(name: str) -> str:
        return f"smolvla_object_{name}"

    def _object_callback(self, message: TFMessage) -> None:
        now = time.monotonic()
        with self._lock:
            for transform in message.transforms:
                name = canonical_object_name(transform.child_frame_id)
                if name is None:
                    continue
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                self._poses[name] = (
                    now,
                    (
                        float(translation.x),
                        float(translation.y),
                        float(translation.z),
                    ),
                    (
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                        float(rotation.w),
                    ),
                )

    def _grasp_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            return
        target = str(status.get("target_object", self._target_object))
        if target not in OBJECT_FRAME_NAMES:
            return
        grasp_success = bool(status.get("grasp_success", False))
        with self._lock:
            if target != self._target_object:
                self._force_update = True
            self._target_object = target
            if grasp_success != self._grasp_success:
                self._force_update = True
            self._grasp_success = grasp_success

    def _request_base_acm(self) -> None:
        if self._base_acm is not None or self._get_pending:
            return
        if not self._get_client.service_is_ready():
            self._get_client.wait_for_service(timeout_sec=0.02)
            return
        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        )
        self._get_pending = True
        future = self._get_client.call_async(request)
        future.add_done_callback(self._base_acm_done)

    def _base_acm_done(self, future) -> None:
        self._get_pending = False
        try:
            response = future.result()
        except Exception as exc:
            self._last_error = f"读取 MoveIt ACM 失败：{exc}"
            self._publish_status("ERROR", self._last_error)
            return
        self._base_acm = copy.deepcopy(response.scene.allowed_collision_matrix)
        self._force_update = True

    def _poses_need_update(self, poses: Dict[str, PoseSample]) -> bool:
        if set(poses) != set(self._last_applied_poses):
            return True
        for name, (_, position, orientation) in poses.items():
            _, old_position, old_orientation = self._last_applied_poses[name]
            position_error = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(position, old_position))
            )
            if position_error >= self.position_threshold:
                return True
            if (
                quaternion_angle(orientation, old_orientation)
                >= self.rotation_threshold
            ):
                return True
        return False

    def _update(self) -> None:
        self._request_base_acm()
        if self._base_acm is None or self._apply_pending:
            return
        if not self._apply_client.service_is_ready():
            self._apply_client.wait_for_service(timeout_sec=0.02)
            return
        now = time.monotonic()
        with self._lock:
            poses = {
                name: sample
                for name, sample in self._poses.items()
                if now - sample[0] <= self.object_timeout
            }
            changed = (
                self._force_update
                or self._target_object != self._last_applied_target
                or self._poses_need_update(poses)
            )
            target = self._target_object
        if len(poses) != len(OBJECT_COLLISIONS):
            self._publish_status(
                "WAITING",
                f"等待物体位姿：{len(poses)}/{len(OBJECT_COLLISIONS)}",
            )
            return
        if not changed:
            return
        try:
            scene = self._build_scene(poses, target)
        except RuntimeError as exc:
            self._last_error = str(exc)
            self._publish_status("ERROR", self._last_error)
            return
        self._attempts += 1
        self._apply_pending = True
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._apply_client.call_async(request)
        future.add_done_callback(
            lambda done: self._apply_done(done, poses, target)
        )

    def _build_scene(
        self,
        poses: Dict[str, PoseSample],
        target: str,
    ) -> PlanningScene:
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        for name, (_, position, orientation) in poses.items():
            scene.world.collision_objects.append(
                self._world_collision(name, position, orientation)
            )
        scene.allowed_collision_matrix = self._target_acm(target)
        return scene

    def _primitive(self, name: str) -> Tuple[SolidPrimitive, float]:
        config = OBJECT_COLLISIONS[name]
        primitive = SolidPrimitive()
        if config["shape"] == "sphere":
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [float(config["radius"])]
            z_offset = float(config["radius"])
        elif config["shape"] == "cylinder":
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [
                float(config["height"]),
                float(config["radius"]),
            ]
            z_offset = float(config["height"]) / 2.0
        else:
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = list(config["size"])
            z_offset = float(config["size"][2]) / 2.0
        return primitive, z_offset

    def _world_collision(
        self, name: str, root_position: Vector3, root_orientation: Quaternion
    ) -> CollisionObject:
        primitive, z_offset = self._primitive(name)
        offset = rotate_vector(root_orientation, (0.0, 0.0, z_offset))
        collision = CollisionObject()
        collision.header.frame_id = self.world_frame
        collision.id = self._object_id(name)
        collision.primitives.append(primitive)
        pose = Pose()
        pose.position.x = root_position[0] + offset[0]
        pose.position.y = root_position[1] + offset[1]
        pose.position.z = root_position[2] + offset[2]
        pose.orientation.x = root_orientation[0]
        pose.orientation.y = root_orientation[1]
        pose.orientation.z = root_orientation[2]
        pose.orientation.w = root_orientation[3]
        collision.primitive_poses.append(pose)
        collision.operation = CollisionObject.ADD
        return collision

    def _target_acm(self, target: str):
        matrix = copy.deepcopy(self._base_acm)
        target_id = self._object_id(target)
        for name in [target_id, *self.touch_links]:
            self._ensure_acm_name(matrix, name)
        index = {
            name: offset for offset, name in enumerate(matrix.entry_names)
        }
        for link in self.touch_links:
            matrix.entry_values[index[target_id]].enabled[index[link]] = True
            matrix.entry_values[index[link]].enabled[index[target_id]] = True
        return matrix

    @staticmethod
    def _ensure_acm_name(matrix, name: str) -> None:
        if name in matrix.entry_names:
            return
        old_size = len(matrix.entry_names)
        for row in matrix.entry_values:
            row.enabled.append(False)
        matrix.entry_names.append(name)
        row = AllowedCollisionEntry()
        row.enabled = [False] * (old_size + 1)
        matrix.entry_values.append(row)

    def _apply_done(
        self,
        future,
        poses: Dict[str, PoseSample],
        target: str,
    ) -> None:
        try:
            response = future.result()
        except Exception as exc:
            self._apply_pending = False
            self._last_error = f"同步动态碰撞体失败：{exc}"
            self._publish_status("ERROR", self._last_error)
            return
        if response is None or not response.success:
            self._verify_applied_state(poses, target)
            return
        self._record_success(poses, target)

    def _verify_applied_state(
        self,
        poses: Dict[str, PoseSample],
        target: str,
    ) -> None:
        if not self._get_client.service_is_ready():
            self._apply_pending = False
            self._last_error = "MoveIt 返回失败且无法核验场景"
            self._publish_status("ERROR", self._last_error)
            return
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        future = self._get_client.call_async(request)
        future.add_done_callback(
            lambda done: self._verify_done(done, poses, target)
        )

    def _verify_done(
        self,
        future,
        poses: Dict[str, PoseSample],
        target: str,
    ) -> None:
        try:
            response = future.result()
            scene = response.scene
        except Exception as exc:
            self._apply_pending = False
            self._last_error = f"核验动态碰撞场景失败：{exc}"
            self._publish_status("ERROR", self._last_error)
            return
        world_ids = {item.id for item in scene.world.collision_objects}
        expected_world = {
            self._object_id(name)
            for name in OBJECT_COLLISIONS
        }
        actual_world = {
            object_id
            for object_id in world_ids
            if object_id.startswith("smolvla_object_")
        }
        if actual_world == expected_world:
            self._record_success(poses, target)
            return
        self._apply_pending = False
        self._last_error = (
            "MoveIt 未形成期望碰撞状态："
            f"world={sorted(actual_world)}"
        )
        self._publish_status("ERROR", self._last_error)

    def _record_success(
        self,
        poses: Dict[str, PoseSample],
        target: str,
    ) -> None:
        self._apply_pending = False
        with self._lock:
            self._last_applied_poses = copy.deepcopy(poses)
            self._last_applied_target = target
            self._force_update = False
            self._last_error = ""
        message = "五个动态物体已同步"
        self._publish_status("READY", message)

    def _reload_callback(self, _request, response):
        with self._lock:
            self._force_update = True
            self._last_applied_poses.clear()
        response.success = True
        response.message = "动态物体碰撞重新同步已触发"
        return response

    def _publish_status(self, phase: str, message: str) -> None:
        with self._lock:
            payload = {
                "phase": phase,
                "ready": phase == "READY",
                "message": message,
                "world_object_count": len(self._poses),
                "target_object": self._target_object,
                "contact_allowed_links": list(self.touch_links),
                "attempts": self._attempts,
                "error": self._last_error,
            }
        status = String()
        status.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._status_pub.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectCollisionSync()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
