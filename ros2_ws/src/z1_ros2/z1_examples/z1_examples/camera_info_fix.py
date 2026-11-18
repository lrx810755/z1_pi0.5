#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo


class CameraInfoFix(Node):
    def __init__(self):
        super().__init__("camera_info_fix")
        # Factory intrinsics from the connected D435i 640x360 RGB profile.
        self.declare_parameter("color_fx", 451.606170654297)
        self.declare_parameter("color_fy", 451.370849609375)
        self.declare_parameter("color_cx", 325.197723388672)
        self.declare_parameter("color_cy", 188.640960693359)
        self.declare_parameter("depth_fx", 387.84747314453125)
        self.declare_parameter("depth_fy", 387.84747314453125)
        self.declare_parameter("depth_cx", 319.447265625)
        self.declare_parameter("depth_cy", 238.202484130859)
        self.declare_parameter("color_input_topic", "/camera/camera/color/camera_info_raw")
        self.declare_parameter("color_output_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("depth_input_topic", "/camera/camera/depth/camera_info_raw")
        self.declare_parameter("depth_output_topic", "/camera/camera/depth/camera_info")

        self.color_k, self.color_p = self._projection("color")
        self.depth_k, self.depth_p = self._projection("depth")

        color_input = self.get_parameter("color_input_topic").value
        color_output = self.get_parameter("color_output_topic").value
        depth_input = self.get_parameter("depth_input_topic").value
        depth_output = self.get_parameter("depth_output_topic").value

        self.color_pub = self.create_publisher(CameraInfo, color_output, 10)
        self.depth_pub = self.create_publisher(CameraInfo, depth_output, 10)
        self.create_subscription(CameraInfo, color_input, self._color_cb, 10)
        self.create_subscription(CameraInfo, depth_input, self._depth_cb, 10)

    def _projection(self, stream: str):
        fx = float(self.get_parameter(f"{stream}_fx").value)
        fy = float(self.get_parameter(f"{stream}_fy").value)
        cx = float(self.get_parameter(f"{stream}_cx").value)
        cy = float(self.get_parameter(f"{stream}_cy").value)
        k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return k, p

    def _fix(self, msg: CameraInfo, k, p) -> CameraInfo:
        fixed = CameraInfo()
        fixed.header = msg.header
        fixed.height = msg.height
        fixed.width = msg.width
        fixed.distortion_model = msg.distortion_model
        fixed.d = list(msg.d)
        fixed.r = list(msg.r) if any(msg.r) else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        fixed.k = list(k)
        fixed.p = list(p)
        fixed.binning_x = msg.binning_x
        fixed.binning_y = msg.binning_y
        fixed.roi = msg.roi
        return fixed

    def _color_cb(self, msg: CameraInfo):
        self.color_pub.publish(self._fix(msg, self.color_k, self.color_p))

    def _depth_cb(self, msg: CameraInfo):
        self.depth_pub.publish(self._fix(msg, self.depth_k, self.depth_p))


def main():
    rclpy.init()
    node = CameraInfoFix()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
