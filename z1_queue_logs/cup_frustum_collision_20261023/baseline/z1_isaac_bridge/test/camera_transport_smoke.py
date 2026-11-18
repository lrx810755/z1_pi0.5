#!/usr/bin/env python3
"""Exercise full-size RGB transport across processes in an isolated ROS domain.

Run after sourcing ROS: python3 test/camera_transport_smoke.py [--profile XML].
No robot topics/services are accessed. Fails on gaps exceeding the recording
window (90 ms of simulated time), even when average arrival frequency is high.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(role):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node('camera_transport_' + role)
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    topics = ('wrist', 'global', 'global_right')
    stamps = {name: [] for name in topics}
    try:
        if role == 'reader':
            for name in topics:
                node.create_subscription(
                    Image, '/transport_test/' + name,
                    lambda m, name=name: stamps[name].append(
                        m.header.stamp.sec + m.header.stamp.nanosec * 1e-9), qos)
            print('READY', flush=True)
            until = time.monotonic() + 15
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=.05)
            result = {}
            for name, values in stamps.items():
                gaps = [b - a for a, b in zip(values, values[1:])]
                result[name] = dict(received=len(values),
                                    max_stamp_gap_s=max(gaps, default=999),
                                    gaps_over_90ms=sum(g > .09 for g in gaps))
            print(json.dumps(result), flush=True)
            return int(any(v['received'] < 100 or v['gaps_over_90ms']
                           for v in result.values()))
        pubs = [node.create_publisher(Image, '/transport_test/' + name, qos)
                for name in topics]
        until = time.monotonic() + 5
        while not all(p.get_subscription_count() for p in pubs):
            if time.monotonic() > until:
                raise RuntimeError('test reader was not discovered')
            rclpy.spin_once(node, timeout_sec=.05)
        messages = []
        for name in topics:
            m = Image()
            m.height, m.width = (360 if name == 'wrist' else 480), 640
            m.encoding, m.step = 'rgb8', 1920
            m.data = bytes(m.height * m.step)
            messages.append(m)
        start = time.monotonic()
        for i in range(300):
            ns = round((1 + i / 60) * 1e9)
            for pub, msg in zip(pubs, messages):
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 10**9)
                pub.publish(msg)
            time.sleep(max(0, start + (i + 1) / 60 - time.monotonic()))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile')
    parser.add_argument('--role', choices=['reader', 'writer'])
    args = parser.parse_args()
    if args.role:
        return worker(args.role)
    env = dict(os.environ, ROS_DOMAIN_ID='91', RMW_IMPLEMENTATION='rmw_fastrtps_cpp')
    for key in ('FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE'):
        env.pop(key, None)
    if args.profile:
        env['FASTRTPS_DEFAULT_PROFILES_FILE'] = str(Path(args.profile).resolve())
    command = [sys.executable, str(Path(__file__).resolve())]
    reader = subprocess.Popen(command + ['--role', 'reader'], env=env,
                              stdout=subprocess.PIPE, text=True)
    try:
        if reader.stdout.readline().strip() != 'READY':
            raise RuntimeError('reader startup failed')
        subprocess.run(command + ['--role', 'writer'], env=env, check=True, timeout=20)
        output, _ = reader.communicate(timeout=20)
        print(output, end='')
        return reader.returncode
    finally:
        if reader.poll() is None:
            reader.terminate()
            reader.wait(timeout=5)


if __name__ == '__main__':
    sys.exit(main())
