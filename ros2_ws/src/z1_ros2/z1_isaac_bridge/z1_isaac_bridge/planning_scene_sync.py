"""Register the Isaac tabletop and target box in the MoveIt planning scene."""

from __future__ import annotations

import json
from typing import Dict, Iterable, Tuple

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import Trigger


BoxSpec = Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]


class PlanningSceneSync(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_planning_scene_sync")
        self.declare_parameter("frame_id", "link00")
        self.declare_parameter("table_center", [0.30, 0.0, -0.025])
        self.declare_parameter("table_size", [0.90, 0.70, 0.05])
        self.declare_parameter("box_center_xy", [0.32, 0.14])
        self.declare_parameter("box_outer_size", 0.16)
        self.declare_parameter("box_wall_thickness", 0.01)
        self.declare_parameter("box_wall_height", 0.10)
        self.declare_parameter("box_bottom_thickness", 0.01)

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.table_center = self._vector_parameter("table_center", 3)
        self.table_size = self._vector_parameter("table_size", 3)
        self.box_center = self._vector_parameter("box_center_xy", 2)
        self.box_outer = float(self.get_parameter("box_outer_size").value)
        self.box_wall = float(
            self.get_parameter("box_wall_thickness").value
        )
        self.box_height = float(
            self.get_parameter("box_wall_height").value
        )
        self.box_bottom = float(
            self.get_parameter("box_bottom_thickness").value
        )

        callback_group = ReentrantCallbackGroup()
        self._client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
            callback_group=callback_group,
        )
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._status_pub = self.create_publisher(
            String, "/smolvla/planning_scene_status", latched_qos
        )
        self.create_service(
            Trigger,
            "/smolvla/reload_collision_scene",
            self._reload_callback,
            callback_group=callback_group,
        )
        self._pending = False
        self._applied = False
        self._attempts = 0
        self._specs = tuple(self._collision_specs())
        self._publish_status("WAITING", "等待 MoveIt PlanningScene 服务")
        self._timer = self.create_timer(
            0.5, self._ensure_scene, callback_group=callback_group
        )

    def _vector_parameter(self, name: str, size: int) -> Tuple[float, ...]:
        values = tuple(
            float(value) for value in self.get_parameter(name).value
        )
        if len(values) != size:
            raise ValueError(f"{name} must contain {size} values")
        return values

    def _collision_specs(self) -> Iterable[BoxSpec]:
        yield "smolvla_table", self.table_center, self.table_size
        bx, by = self.box_center
        outer = self.box_outer
        wall = self.box_wall
        height = self.box_height
        bottom = self.box_bottom
        wall_z = bottom + height / 2.0
        wall_offset = (outer - wall) / 2.0
        yield (
            "smolvla_box_bottom",
            (bx, by, bottom / 2.0),
            (outer, outer, bottom),
        )
        yield (
            "smolvla_box_left",
            (bx - wall_offset, by, wall_z),
            (wall, outer, height),
        )
        yield (
            "smolvla_box_right",
            (bx + wall_offset, by, wall_z),
            (wall, outer, height),
        )
        yield (
            "smolvla_box_front",
            (bx, by - wall_offset, wall_z),
            (outer - 2.0 * wall, wall, height),
        )
        yield (
            "smolvla_box_back",
            (bx, by + wall_offset, wall_z),
            (outer - 2.0 * wall, wall, height),
        )

    def _collision_object(self, spec: BoxSpec) -> CollisionObject:
        object_id, center, size = spec
        collision = CollisionObject()
        collision.header.frame_id = self.frame_id
        collision.id = object_id
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(size)
        pose = Pose()
        pose.position.x = center[0]
        pose.position.y = center[1]
        pose.position.z = center[2]
        pose.orientation.w = 1.0
        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose)
        collision.operation = CollisionObject.ADD
        return collision

    def _scene_request(self) -> ApplyPlanningScene.Request:
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = [
            self._collision_object(spec) for spec in self._specs
        ]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        return request

    def _ensure_scene(self) -> None:
        if self._applied or self._pending:
            return
        if not self._client.service_is_ready():
            self._client.wait_for_service(timeout_sec=0.05)
            return
        self._attempts += 1
        self._pending = True
        self._publish_status("APPLYING", "正在注册桌面与蓝盒碰撞体")
        future = self._client.call_async(self._scene_request())
        future.add_done_callback(self._apply_done)

    def _apply_done(self, future) -> None:
        self._pending = False
        try:
            response = future.result()
        except Exception as exc:
            self._publish_status("ERROR", f"PlanningScene 调用失败：{exc}")
            return
        if response is None or not response.success:
            self._publish_status("ERROR", "MoveIt 拒绝碰撞场景")
            return
        self._applied = True
        message = f"已注册 {len(self._specs)} 个碰撞体：桌面 + 蓝盒底板和四壁"
        self._publish_status("READY", message)
        self.get_logger().info(message)

    def _reload_callback(self, _request, response):
        self._applied = False
        self._pending = False
        self._ensure_scene()
        response.success = True
        response.message = "碰撞场景重新同步已触发"
        return response

    def _publish_status(self, phase: str, message: str) -> None:
        payload: Dict[str, object] = {
            "phase": phase,
            "ready": phase == "READY",
            "message": message,
            "frame_id": self.frame_id,
            "object_count": len(self._specs) if hasattr(self, "_specs") else 0,
            "object_ids": [spec[0] for spec in getattr(self, "_specs", ())],
            "attempts": self._attempts if hasattr(self, "_attempts") else 0,
        }
        status = String()
        status.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._status_pub.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningSceneSync()
    executor = MultiThreadedExecutor(num_threads=2)
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
