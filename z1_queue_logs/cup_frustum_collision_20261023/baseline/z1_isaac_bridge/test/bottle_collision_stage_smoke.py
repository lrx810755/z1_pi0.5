"""Manual integration check in an isolated ROS_DOMAIN_ID; no robot commands.

Pass the running MoveGroup's parameter files as positional arguments.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from std_srvs.srv import SetBool
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents
from z1_isaac_bridge.object_collision_sync import ObjectCollisionSync, OBJECT_COLLISIONS


def main():
    assert os.environ.get('ROS_DOMAIN_ID') == '88', 'Run only in isolated domain 88'
    output = Path('/home/lp531/smolvla_eval/bottle_collision_stages_20261005')
    output.mkdir(parents=True, exist_ok=True)
    command = ['/opt/ros/humble/lib/moveit_ros_move_group/move_group', '--ros-args']
    for parameter_file in sys.argv[1:]:
        command += ['--params-file', parameter_file]
    command += ['-p', 'use_sim_time:=false']
    log = (output / 'isolated_move_group.log').open('w')
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    rclpy.init()
    sync = ObjectCollisionSync()
    probe = Node('bottle_collision_stage_probe')
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(sync)
    executor.add_node(probe)
    worker = threading.Thread(target=executor.spin, daemon=True)
    worker.start()
    truth = probe.create_publisher(TFMessage, '/isaac/object_transforms', 10)
    status = probe.create_publisher(String, '/smolvla/status', 10)

    def publish_scene():
        message = TFMessage()
        for i, name in enumerate(OBJECT_COLLISIONS):
            transform = TransformStamped()
            transform.header.frame_id = 'world'
            transform.child_frame_id = name
            transform.transform.translation.x = 0.4 + i * 0.08
            transform.transform.translation.y = -0.2
            transform.transform.rotation.w = 1.0
            message.transforms.append(transform)
        truth.publish(message)
        status.publish(String(data=json.dumps({'target_object': 'beverage_bottle'})))

    timer = probe.create_timer(0.05, publish_scene)
    set_mode = probe.create_client(SetBool, '/smolvla/set_bottle_finger_collision')
    get_scene = probe.create_client(GetPlanningScene, '/get_planning_scene')

    def call(client, request):
        future = client.call_async(request)
        deadline = time.monotonic() + 9
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert future.done(), 'Service response timeout'
        return future.result()

    results = []
    try:
        assert get_scene.wait_for_service(timeout_sec=20), 'MoveIt did not start'
        assert set_mode.wait_for_service(timeout_sec=5)
        deadline = time.monotonic() + 10
        while sync._applied_bottle_finger_contact is None and time.monotonic() < deadline:
            time.sleep(0.05)
        for enabled in (False, True, False):
            response = call(set_mode, SetBool.Request(data=enabled))
            assert response.success, response.message
            # Allow periodic updates to run so an accidental overwrite is caught.
            time.sleep(0.5)
            request = GetPlanningScene.Request()
            request.components.components = (PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
                                             | PlanningSceneComponents.WORLD_OBJECT_NAMES)
            scene = call(get_scene, request).scene
            matrix = scene.allowed_collision_matrix
            print('mode', enabled, 'world', [item.id for item in scene.world.collision_objects], flush=True)
            if 'smolvla_object_beverage_bottle' not in matrix.entry_names:
                print('ACM readback:', list(matrix.entry_names), list(matrix.default_entry_names),
                      'pending:', list(sync._pending_acm.entry_names), flush=True)
            permissions = {}
            if 'smolvla_object_beverage_bottle' in matrix.entry_names:
                index = matrix.entry_names.index('smolvla_object_beverage_bottle')
                permissions = dict(zip(matrix.entry_names, matrix.entry_values[index].enabled))
            permitted = sorted(name for name, value in permissions.items() if value)
            assert permitted == (['gripperMover', 'gripperStator'] if enabled else []), permitted
            results.append({'finger_contact': enabled, 'allowed_links': permitted})
        (output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results))
    finally:
        probe.destroy_timer(timer)
        executor.shutdown()
        worker.join(timeout=5)
        sync.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == '__main__':
    main()
