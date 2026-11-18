#!/usr/bin/env python3

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


HOST = os.environ.get("Z1_MOVEIT_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("Z1_MOVEIT_BRIDGE_PORT", "8766"))
MOVE_GROUP_ACTION = os.environ.get("Z1_MOVE_GROUP_ACTION", "/move_action")
GROUP_NAME = os.environ.get("Z1_MOVEIT_GROUP", "z1_arm")
EE_LINK = os.environ.get("Z1_MOVEIT_EE_LINK", "link06")
BASE_FRAME = os.environ.get("Z1_MOVEIT_BASE_FRAME", "link00")
PLANNING_TIME_S = float(os.environ.get("Z1_MOVEIT_PLANNING_TIME_S", "8.0"))
GOAL_BOX_SIZE_M = float(os.environ.get("Z1_MOVEIT_GOAL_BOX_SIZE_M", "0.015"))
ORIENTATION_TOLERANCE_RAD = float(os.environ.get("Z1_MOVEIT_ORIENTATION_TOLERANCE_RAD", "0.35"))
ORIENTATION_WEIGHT = float(os.environ.get("Z1_MOVEIT_ORIENTATION_WEIGHT", "0.0"))
ACTION_WAIT_TIMEOUT_S = float(os.environ.get("Z1_MOVEIT_ACTION_WAIT_TIMEOUT_S", "15.0"))


class Z1MoveGroupBridge(Node):
    def __init__(self):
        super().__init__("z1_moveit_http_bridge")
        self.client = ActionClient(self, MoveGroup, MOVE_GROUP_ACTION)
        self.lock = threading.Lock()

    def wait_until_ready(self):
        if not self.client.wait_for_server(timeout_sec=ACTION_WAIT_TIMEOUT_S):
            raise RuntimeError(f"MoveGroup action server 不可用: {MOVE_GROUP_ACTION}")

    def plan_execute(self, payload):
        with self.lock:
            start = time.monotonic()
            goal = self._goal_from_payload(payload)
            send_future = self.client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("MoveGroup 拒绝了规划请求")

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()
            if result is None:
                raise RuntimeError("MoveGroup 未返回执行结果")

            action_result = result.result
            code = action_result.error_code.val
            if code != MoveItErrorCodes.SUCCESS:
                raise RuntimeError(f"MoveGroup 规划或执行失败，error_code={code}")

            elapsed = time.monotonic() - start
            return {
                "success": True,
                "phase": payload.get("phase", ""),
                "group": payload.get("group") or GROUP_NAME,
                "end_effector_link": payload.get("end_effector_link") or EE_LINK,
                "frame_id": payload.get("frame_id") or BASE_FRAME,
                "planning_time_s": float(action_result.planning_time),
                "response_time_s": elapsed,
                "trajectory": self._trajectory_to_json(action_result.planned_trajectory.joint_trajectory),
            }

    def _trajectory_to_json(self, trajectory):
        points = []
        for point in trajectory.points:
            points.append(
                {
                    "positions": [float(value) for value in point.positions],
                    "velocities": [float(value) for value in point.velocities],
                    "accelerations": [float(value) for value in point.accelerations],
                    "time_from_start_s": float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9,
                }
            )
        return {
            "joint_names": list(trajectory.joint_names),
            "points": points,
            "point_count": len(points),
        }

    def _goal_from_payload(self, payload):
        position = payload.get("position_xyz_m")
        orientation = payload.get("orientation_xyzw")
        if not isinstance(position, list) or len(position) != 3:
            raise RuntimeError("position_xyz_m 必须是长度为 3 的数组")
        if ORIENTATION_WEIGHT > 0.0 and (not isinstance(orientation, list) or len(orientation) != 4):
            raise RuntimeError("orientation_xyzw 必须是长度为 4 的数组")

        group = payload.get("group") or GROUP_NAME
        ee_link = payload.get("end_effector_link") or EE_LINK
        frame_id = payload.get("frame_id") or BASE_FRAME
        speed_scale = max(min(float(payload.get("speed_scale", 0.5)), 1.0), 0.05)

        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = group
        request.num_planning_attempts = 5
        request.allowed_planning_time = PLANNING_TIME_S
        request.max_velocity_scaling_factor = speed_scale
        request.max_acceleration_scaling_factor = speed_scale
        self._apply_start_state(request, payload.get("start_joint_state"))
        request.goal_constraints = [self._constraints(frame_id, ee_link, position, orientation)]

        options = goal.planning_options
        options.plan_only = True
        options.look_around = False
        options.replan = False
        options.replan_attempts = 0
        options.replan_delay = 0.0
        options.planning_scene_diff.is_diff = True
        options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def _apply_start_state(self, request, start_joint_state):
        if not isinstance(start_joint_state, dict):
            request.start_state.is_diff = True
            return

        joint_names = start_joint_state.get("joint_names")
        positions = start_joint_state.get("positions")
        if not isinstance(joint_names, list) or not isinstance(positions, list) or len(joint_names) != len(positions):
            request.start_state.is_diff = True
            return

        joint_state = JointState()
        joint_state.name = [str(name) for name in joint_names]
        joint_state.position = [float(value) for value in positions]
        request.start_state.joint_state = joint_state
        request.start_state.is_diff = False

    def _constraints(self, frame_id, link_name, position, orientation):
        constraints = Constraints()
        constraints.name = "z1_pose_goal"

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [GOAL_BOX_SIZE_M]

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.w = 1.0

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = frame_id
        position_constraint.link_name = link_name
        position_constraint.constraint_region.primitives.append(sphere)
        position_constraint.constraint_region.primitive_poses.append(pose)
        position_constraint.weight = 1.0

        constraints.position_constraints.append(position_constraint)
        if ORIENTATION_WEIGHT > 0.0:
            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = frame_id
            orientation_constraint.link_name = link_name
            orientation_constraint.orientation.x = float(orientation[0])
            orientation_constraint.orientation.y = float(orientation[1])
            orientation_constraint.orientation.z = float(orientation[2])
            orientation_constraint.orientation.w = float(orientation[3])
            orientation_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE_RAD
            orientation_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE_RAD
            orientation_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE_RAD
            orientation_constraint.weight = ORIENTATION_WEIGHT
            constraints.orientation_constraints.append(orientation_constraint)
        return constraints


bridge_node = None


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/plan_execute":
            self.send_json({"success": False, "error": "not found"}, status=404)
            return
        try:
            payload = self.read_json()
            if bridge_node is None:
                raise RuntimeError("MoveIt bridge 未初始化")
            result = bridge_node.plan_execute(payload)
            self.send_json(result)
        except Exception as exc:
            print(f"[z1_moveit_bridge] request failed: {exc}", flush=True)
            self.send_json({"success": False, "error": str(exc)}, status=400)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main():
    global bridge_node
    rclpy.init()
    bridge_node = Z1MoveGroupBridge()

    bridge_node.wait_until_ready()
    print(f"[z1_moveit_bridge] MoveGroup ready: action={MOVE_GROUP_ACTION} group={GROUP_NAME} ee={EE_LINK} frame={BASE_FRAME}")

    server = ThreadingHTTPServer((HOST, PORT), BridgeHandler)
    print(f"[z1_moveit_bridge] listening on http://{HOST}:{PORT}/plan_execute")
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        bridge_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
