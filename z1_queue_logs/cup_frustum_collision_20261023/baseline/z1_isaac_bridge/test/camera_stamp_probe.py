#!/usr/bin/env python3
"""Read camera header stamps and /clock without retaining image payloads."""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=60)
    args = parser.parse_args()
    rclpy.init()
    node = Node('camera_stamp_probe')
    rows = {k: [] for k in ('wrist', 'global', 'vla_wrist')}
    clocks = []
    recording = {}
    gap_events = []

    def recording_status(message):
        recording.clear()
        recording.update(json.loads(message.data))

    node.create_subscription(String, '/smolvla/recording_status', recording_status, 10)
    node.create_subscription(
        Clock, '/clock', lambda m: clocks.append(
            m.clock.sec + m.clock.nanosec * 1e-9), qos_profile_sensor_data)

    def receive(name, message):
        stamp = message.header.stamp
        seconds = stamp.sec + stamp.nanosec * 1e-9
        if rows[name] and seconds - rows[name][-1][0] > .09:
            gap_events.append({
                'camera': name, 'before': rows[name][-1][0], 'after': seconds,
                'recording_phase': recording.get('phase'),
                'episode_index': recording.get('episode_index'),
                'recorder_sync_failure': recording.get('sync_failure'),
            })
        rows[name].append((seconds, time.monotonic()))

    for name, topic in (
        ('wrist', '/camera/camera/color/image_raw'),
        ('global', '/global_camera/color/image_raw'),
        ('vla_wrist', '/vla_wrist_camera/color/image_raw'),
    ):
        node.create_subscription(Image, topic,
                                 lambda m, name=name: receive(name, m),
                                 qos_profile_sensor_data)
    try:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
        report = {'duration_wall_s': args.duration, 'cameras': {}}
        for name, values in rows.items():
            stamp_gaps = [b[0] - a[0] for a, b in zip(values, values[1:])]
            arrival_gaps = [b[1] - a[1] for a, b in zip(values, values[1:])]
            report['cameras'][name] = {
                'received': len(values),
                'max_stamp_gap_s': max(stamp_gaps, default=None),
                'max_arrival_gap_s': max(arrival_gaps, default=None),
                'stamp_gaps_over_90ms': sum(g > .09 for g in stamp_gaps),
                'nonmonotonic_stamps': sum(g <= 0 for g in stamp_gaps),
            }
        report['clock_backwards'] = sum(b < a for a, b in zip(clocks, clocks[1:]))
        report['gap_events'] = gap_events
        print(json.dumps(report, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
