"""Exercise real tf2 buffers across a simulation-clock restart."""
from collections import deque
from types import SimpleNamespace
import pytest
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
import tf2_ros
from z1_isaac_bridge.grasp_evaluator import GraspEvaluator

def transform(stamp,parent='world',child='gripperStator',x=.5):
 t=TransformStamped();t.header.frame_id=parent;t.child_frame_id=child
 t.header.stamp=Time(seconds=stamp).to_msg();t.transform.translation.x=x;t.transform.rotation.w=1.
 return t

def evaluator():
 e=object.__new__(GraspEvaluator);e.tf_buffer=tf2_ros.Buffer()
 e.world_frame='world';e.gripper_frame='gripperStator';e.target_object='beverage_bottle'
 e.last_clock=0.;e.last_tf_fallback_warning=0.;e.max_gripper_tf_age=.1
 e.object_poses={};e.baseline_z={};e.relative_history=deque();e.gripper_history=deque()
 e.gripper_position=None;e.get_logger=lambda:SimpleNamespace(warning=lambda _:None)
 e._publish_outputs=lambda *args,**kwargs:None;e._now=lambda:1.
 return e

def test_clock_rewind_discards_previous_simulation_dynamic_tf():
 e=evaluator();e.last_clock=82000.
 e.tf_buffer.set_transform_static(transform(0,'world','base',x=0.),'static')
 e.tf_buffer.set_transform(transform(82000,'base','gripperStator',x=.12),'old_sim')
 GraspEvaluator._evaluate(e)
 assert not e.tf_buffer.can_transform('world','gripperStator',Time())
 e.tf_buffer.set_transform(transform(1,'base','gripperStator',x=.55),'new_sim')
 assert e._gripper_pose(1)[0]==pytest.approx((.55,0,0))

def test_fallback_rejects_a_transform_from_the_previous_simulation_future():
 e=evaluator();e.tf_buffer.set_transform(transform(82000,x=.12),'old_sim')
 assert e._gripper_pose(1) is None

def test_fallback_rejects_stale_transform():
 e=evaluator();e.tf_buffer.set_transform(transform(1),'sim')
 assert e._gripper_pose(1.5) is None

def test_fallback_allows_recent_transform_within_sensor_delay():
 e=evaluator();e.tf_buffer.set_transform(transform(1),'sim')
 assert e._gripper_pose(1.03)[0]==pytest.approx((.5,0,0))
