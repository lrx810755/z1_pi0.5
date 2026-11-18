#!/usr/bin/env python3
"""Publish a short synthetic successful episode for recorder smoke tests."""

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger


JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "jointGripper",
]


class SyntheticEpisode(Node):
    def __init__(self):
        super().__init__("synthetic_episode_source")
        self.wrist_pub = self.create_publisher(
            Image, "/camera/camera/color/image_raw", 10
        )
        self.global_pub = self.create_publisher(
            Image, "/global_camera/color/image_raw", 10
        )
        self.state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.action_pub = self.create_publisher(
            JointState, "/isaac_joint_commands", 10
        )
        self.status_pub = self.create_publisher(String, "/smolvla/status", 10)
        self.start_client = self.create_client(
            Trigger, "/smolvla/recording/start"
        )
        self.save_client = self.create_client(
            Trigger, "/smolvla/recording/save"
        )
        self.success = False
        self.step = 0

    def publish(self):
        stamp = self.get_clock().now().to_msg()
        base = np.zeros((48, 64, 3), dtype=np.uint8)
        base[:, :, 0] = (self.step * 7) % 255
        base[:, :, 1] = 90
        base[:, :, 2] = 180
        for publisher, image in (
            (self.wrist_pub, base),
            (self.global_pub, np.flip(base, axis=1).copy()),
        ):
            message = Image()
            message.header.stamp = stamp
            message.height = image.shape[0]
            message.width = image.shape[1]
            message.encoding = "rgb8"
            message.step = image.shape[1] * 3
            message.data = image.tobytes()
            publisher.publish(message)

        positions = [0.01 * self.step] * 6 + [-0.2]
        state = JointState()
        state.header.stamp = stamp
        state.name = JOINTS
        state.position = positions
        self.state_pub.publish(state)
        action = JointState()
        action.header.stamp = stamp
        action.name = JOINTS
        action.position = [value + 0.001 for value in positions]
        self.action_pub.publish(action)
        status = String()
        status.data = json.dumps(
            {
                "phase": "SUCCESS" if self.success else "APPROACHING",
                "target_object": "apple",
                "grasp_success": self.step > 5,
                "task_success": self.success,
            }
        )
        self.status_pub.publish(status)
        self.step += 1


def publish_for(node, duration):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        node.publish()
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(0.05)


def call(node, client, name):
    if not client.wait_for_service(timeout_sec=3.0):
        raise RuntimeError(f"service unavailable: {name}")
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    response = future.result()
    if response is None or not response.success:
        raise RuntimeError(
            response.message if response is not None else f"{name} timed out"
        )
    print(response.message, flush=True)


def main():
    rclpy.init()
    node = SyntheticEpisode()
    try:
        publish_for(node, 0.8)
        call(node, node.start_client, "start")
        publish_for(node, 1.5)
        node.success = True
        publish_for(node, 0.5)
        call(node, node.save_client, "save")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
