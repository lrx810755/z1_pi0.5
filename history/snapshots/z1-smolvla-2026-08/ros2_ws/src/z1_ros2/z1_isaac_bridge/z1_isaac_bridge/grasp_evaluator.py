"""Judge simulated grasp and placement success from Isaac ground-truth poses."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import re
import time
from typing import Deque, Dict, Optional, Sequence, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
import tf2_ros

from .grasp_config import (
    GRASP_CONFIG,
    GRASP_ENVELOPE,
    OBJECT_GRASP_PROFILES,
    inside_grasp_envelope,
    object_loaded_grasp_is_valid,
    target_bilateral_contact,
)


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # xyzw

OBJECT_FRAME_NAMES = {
    "cup": "Cup",
    "apple": "Apple",
    "peach": "Peach",
    "mouse": "Mouse",
    "beverage_bottle": "BeverageBottle",
}


@dataclass
class PoseSample:
    stamp: float
    position: Vector3
    orientation: Quaternion
    speed: float = 0.0


def vector_norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    return (-q[0], -q[1], -q[2], q[3])


def quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector(q: Quaternion, value: Vector3) -> Vector3:
    rotated = quaternion_multiply(
        quaternion_multiply(q, (value[0], value[1], value[2], 0.0)),
        quaternion_conjugate(q),
    )
    return rotated[0], rotated[1], rotated[2]


def quaternion_angle(a: Quaternion, b: Quaternion) -> float:
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
    return 2.0 * math.acos(dot)


def canonical_object_name(frame_id: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]", "", frame_id.lower())
    for name, frame_name in OBJECT_FRAME_NAMES.items():
        if normalized.endswith(re.sub(r"[^a-z0-9]", "", frame_name.lower())):
            return name
    return None


def physical_grasp_predicate(
    *,
    gripper_obstructed: bool,
    inside_envelope: bool,
    object_clear_of_table: bool,
    stable_relative_pose: bool,
    object_touching_table: bool,
    finger_touching_table: bool,
    bilateral_contact: bool,
    contact_report_available: bool,
) -> bool:
    """Return true only for a loaded physical lift clear of the table.

    Bilateral contact is required when Isaac supplies contact reports. On the
    compatibility fallback path, obstruction and the two-finger envelope are
    explicitly treated as estimates rather than reported physical contact.
    """
    contact_ok = bilateral_contact if contact_report_available else True
    return (
        gripper_obstructed
        and inside_envelope
        and object_clear_of_table
        and stable_relative_pose
        and not object_touching_table
        and not finger_touching_table
        and contact_ok
    )


def physical_placement_predicate(
    *,
    physical_grasp_was_verified: bool,
    release_sequence_complete: bool,
    released: bool,
    inside_box: bool,
    object_stationary: bool,
    failure_reason: str,
) -> bool:
    """Accept placement after a verified physical transport and release."""
    return (
        physical_grasp_was_verified
        and release_sequence_complete
        and released
        and inside_box
        and object_stationary
        and not failure_reason
    )


class GraspEvaluator(Node):
    """Publish deterministic labels for grasp, transport, and placement phases."""

    def __init__(self) -> None:
        super().__init__("smolvla_grasp_evaluator")
        self.declare_parameter("target_object", "apple")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("gripper_frame", "gripperStator")
        self.declare_parameter("gripper_joint", "jointGripper")
        # URDF geometry closes toward 0 rad and opens toward the -pi/2 limit.
        self.declare_parameter("closed_threshold_rad", -0.75)
        self.declare_parameter("released_threshold_rad", -0.80)
        lift_config = GRASP_CONFIG["physical_test_lift"]
        contact_config = GRASP_CONFIG["contact_report"]
        gripper_config = GRASP_CONFIG["gripper"]
        table_config = GRASP_CONFIG["table"]
        # Deprecated compatibility/debug radius.  It is never used to prove
        # that an object lies between the fingers.
        self.declare_parameter("capture_radius_m", 0.20)
        self.declare_parameter(
            "relative_position_tolerance_m",
            float(lift_config["relative_position_tolerance_m"]),
        )
        self.declare_parameter(
            "relative_rotation_tolerance_deg",
            float(lift_config["relative_rotation_tolerance_deg"]),
        )
        self.declare_parameter(
            "grasp_stable_duration_s",
            float(lift_config["stable_duration_s"]),
        )
        self.declare_parameter(
            "gripper_obstruction_window_s",
            float(gripper_config["obstruction_window_s"]),
        )
        self.declare_parameter(
            "gripper_obstruction_max_motion_rad",
            float(gripper_config["obstruction_max_motion_rad"]),
        )
        self.declare_parameter(
            "gripper_closed_target_rad", 0.0
        )
        self.declare_parameter(
            "gripper_mechanical_limit_tolerance_rad",
            float(gripper_config["mechanical_limit_tolerance_rad"]),
        )
        self.declare_parameter(
            "table_clearance_m", float(table_config["clearance_m"])
        )
        self.declare_parameter(
            "contact_loss_grace_s",
            float(contact_config["loss_grace_s"]),
        )
        self.declare_parameter("placement_stable_duration_s", 0.50)
        self.declare_parameter("placement_speed_threshold_m_s", 0.03)
        self.declare_parameter("box_center_x_m", 0.32)
        self.declare_parameter("box_center_y_m", 0.14)
        self.declare_parameter("box_inner_half_extent_m", 0.045)
        self.declare_parameter("box_min_z_m", 0.0)
        self.declare_parameter("box_max_z_m", 0.12)

        self.target_object = str(self.get_parameter("target_object").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.gripper_frame = str(self.get_parameter("gripper_frame").value)
        self.gripper_joint = str(self.get_parameter("gripper_joint").value)
        self.closed_threshold = float(self.get_parameter("closed_threshold_rad").value)
        self.released_threshold = float(self.get_parameter("released_threshold_rad").value)
        self.capture_radius = float(self.get_parameter("capture_radius_m").value)
        self.relative_position_tolerance = float(
            self.get_parameter("relative_position_tolerance_m").value
        )
        self.relative_rotation_tolerance = math.radians(
            float(self.get_parameter("relative_rotation_tolerance_deg").value)
        )
        self.grasp_stable_duration = float(
            self.get_parameter("grasp_stable_duration_s").value
        )
        self.gripper_obstruction_window = float(
            self.get_parameter("gripper_obstruction_window_s").value
        )
        self.gripper_obstruction_max_motion = float(
            self.get_parameter("gripper_obstruction_max_motion_rad").value
        )
        self.gripper_closed_target = float(
            self.get_parameter("gripper_closed_target_rad").value
        )
        self.gripper_mechanical_limit_tolerance = float(
            self.get_parameter(
                "gripper_mechanical_limit_tolerance_rad"
            ).value
        )
        self.table_clearance = float(
            self.get_parameter("table_clearance_m").value
        )
        self.contact_loss_grace = max(
            0.0,
            float(self.get_parameter("contact_loss_grace_s").value),
        )
        self.placement_stable_duration = float(
            self.get_parameter("placement_stable_duration_s").value
        )
        self.placement_speed_threshold = float(
            self.get_parameter("placement_speed_threshold_m_s").value
        )
        self.box_center = (
            float(self.get_parameter("box_center_x_m").value),
            float(self.get_parameter("box_center_y_m").value),
        )
        self.box_half_extent = float(
            self.get_parameter("box_inner_half_extent_m").value
        )
        self.box_min_z = float(self.get_parameter("box_min_z_m").value)
        self.box_max_z = float(self.get_parameter("box_max_z_m").value)

        if self.target_object not in OBJECT_FRAME_NAMES:
            raise ValueError(f"Unknown target_object: {self.target_object}")

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.object_poses: Dict[str, PoseSample] = {}
        self.baseline_z: Dict[str, float] = {}
        self.gripper_position: Optional[float] = None
        self.relative_history: Deque[Tuple[float, Vector3, Quaternion]] = deque()
        self.gripper_history: Deque[Tuple[float, float]] = deque()
        self.placement_since: Optional[float] = None
        self.task_success = False
        self.was_grasped = False
        self.physical_grasp_ok = False
        self.last_target_bilateral_contact_time: Optional[float] = None
        self.grasp_control: Dict[str, object] = {}
        self.scene_grasp: Dict[str, object] = {
            "frozen": False,
            "contact_report_available": False,
        }
        self.failure_reason = ""
        self.last_clock = 0.0
        self.last_tf_fallback_warning = 0.0

        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.grasp_pub = self.create_publisher(Bool, "/smolvla/grasp_success", latched_qos)
        self.physical_grasp_pub = self.create_publisher(
            Bool, "/smolvla/physical_grasp_ok", latched_qos
        )
        self.task_pub = self.create_publisher(Bool, "/smolvla/task_success", latched_qos)
        self.phase_pub = self.create_publisher(String, "/smolvla/task_phase", latched_qos)
        self.object_pub = self.create_publisher(String, "/smolvla/grasped_object", latched_qos)
        self.status_pub = self.create_publisher(String, "/smolvla/status", 10)

        self.create_subscription(JointState, "/joint_states", self._joint_callback, 20)
        self.create_subscription(
            TFMessage, "/isaac/object_transforms", self._object_callback, 20
        )
        self.create_subscription(String, "/smolvla/target_object", self._target_callback, 10)
        self.create_subscription(
            String,
            "/smolvla/grasp_control_state",
            self._grasp_control_callback,
            latched_qos,
        )
        self.create_subscription(
            String,
            "/smolvla/scene_grasp_status",
            self._scene_grasp_callback,
            latched_qos,
        )
        self.create_service(Trigger, "/smolvla/reset_evaluator", self._reset_service)
        self.create_timer(1.0 / 30.0, self._evaluate)

        self.get_logger().info(
            f"Grasp evaluator ready; target={self.target_object}, "
            "truth=/isaac/object_transforms"
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _reset_state(self, clear_baselines: bool = False) -> None:
        self.relative_history.clear()
        self.gripper_history.clear()
        self.placement_since = None
        self.task_success = False
        self.was_grasped = False
        self.physical_grasp_ok = False
        self.last_target_bilateral_contact_time = None
        self.grasp_control = {}
        self.failure_reason = ""
        if clear_baselines:
            self.baseline_z.clear()

    def _reset_service(self, _request, response):
        self._reset_state(clear_baselines=True)
        response.success = True
        response.message = "Grasp evaluator reset"
        return response

    def _target_callback(self, message: String) -> None:
        target = message.data.strip().lower()
        if target not in OBJECT_FRAME_NAMES:
            self.get_logger().error(
                f"Ignoring unknown target {target!r}; choose {sorted(OBJECT_FRAME_NAMES)}"
            )
            return
        if target != self.target_object:
            self.target_object = target
            self._reset_state(clear_baselines=False)
            self.get_logger().info(f"Target object changed to {target}")

    def _joint_callback(self, message: JointState) -> None:
        try:
            index = message.name.index(self.gripper_joint)
        except ValueError:
            return
        if index < len(message.position):
            self.gripper_position = float(message.position[index])
            now = self._now()
            self.gripper_history.append((now, self.gripper_position))
            cutoff = now - max(1.0, self.gripper_obstruction_window * 2.0)
            while self.gripper_history and self.gripper_history[0][0] < cutoff:
                self.gripper_history.popleft()

    def _grasp_control_callback(self, message: String) -> None:
        try:
            control = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(control, dict):
            self.grasp_control = control

    def _scene_grasp_callback(self, message: String) -> None:
        try:
            scene = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not isinstance(scene, dict):
            return
        self.scene_grasp = scene

    def _object_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            name = canonical_object_name(transform.child_frame_id)
            if name is None:
                continue
            stamp = (
                float(transform.header.stamp.sec)
                + float(transform.header.stamp.nanosec) * 1.0e-9
            )
            if stamp <= 0.0:
                stamp = self._now()
            position = (
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            )
            orientation = (
                float(transform.transform.rotation.x),
                float(transform.transform.rotation.y),
                float(transform.transform.rotation.z),
                float(transform.transform.rotation.w),
            )
            previous = self.object_poses.get(name)
            speed = 0.0
            if previous is not None and stamp > previous.stamp + 1.0e-6:
                speed = vector_norm(
                    tuple(a - b for a, b in zip(position, previous.position))
                ) / (stamp - previous.stamp)
            self.object_poses[name] = PoseSample(stamp, position, orientation, speed)
            self.baseline_z.setdefault(name, position[2])

    def _gripper_pose(
        self, object_stamp: Optional[float] = None
    ) -> Optional[Tuple[Vector3, Quaternion]]:
        stamped_query = object_stamp is not None and object_stamp > 0.0
        query_time = (
            Time(nanoseconds=max(0, int(object_stamp * 1.0e9)))
            if stamped_query
            else Time()
        )
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.gripper_frame, query_time
            ).transform
        except tf2_ros.TransformException as stamped_error:
            if not stamped_query:
                return None
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame, self.gripper_frame, Time()
                ).transform
            except tf2_ros.TransformException:
                return None
            monotonic_now = time.monotonic()
            if monotonic_now - self.last_tf_fallback_warning >= 5.0:
                self.last_tf_fallback_warning = monotonic_now
                self.get_logger().warning(
                    "Object-stamped gripper TF unavailable; falling back to "
                    f"latest transform: {stamped_error}"
                )
        return (
            (
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ),
            (
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            ),
        )

    def _relative_pose(
        self,
        gripper: Tuple[Vector3, Quaternion],
        obj: PoseSample,
    ) -> Tuple[Vector3, Quaternion]:
        gripper_position, gripper_orientation = gripper
        inverse = quaternion_conjugate(gripper_orientation)
        offset = tuple(a - b for a, b in zip(obj.position, gripper_position))
        return (
            rotate_vector(inverse, offset),
            quaternion_multiply(inverse, obj.orientation),
        )

    def _relative_pose_stable(
        self,
        now: float,
        duration: Optional[float] = None,
        position_tolerance: Optional[float] = None,
        rotation_tolerance: Optional[float] = None,
    ) -> bool:
        active_duration = (
            self.grasp_stable_duration if duration is None else float(duration)
        )
        cutoff = now - active_duration
        while self.relative_history and self.relative_history[0][0] < cutoff:
            self.relative_history.popleft()
        if len(self.relative_history) < 2:
            return False
        if self.relative_history[-1][0] - self.relative_history[0][0] < (
            active_duration * 0.9
        ):
            return False
        reference_position = self.relative_history[0][1]
        reference_orientation = self.relative_history[0][2]
        position_error = max(
            vector_norm(tuple(a - b for a, b in zip(position, reference_position)))
            for _, position, _ in self.relative_history
        )
        rotation_error = max(
            quaternion_angle(orientation, reference_orientation)
            for _, _, orientation in self.relative_history
        )
        return (
            position_error
            <= (
                self.relative_position_tolerance
                if position_tolerance is None
                else float(position_tolerance)
            )
            and rotation_error
            <= (
                self.relative_rotation_tolerance
                if rotation_tolerance is None
                else float(rotation_tolerance)
            )
        )

    def _inside_success_volume(self, position: Vector3) -> bool:
        return (
            abs(position[0] - self.box_center[0]) <= self.box_half_extent
            and abs(position[1] - self.box_center[1]) <= self.box_half_extent
            and self.box_min_z <= position[2] <= self.box_max_z
        )

    def _gripper_obstructed_estimate(self, now: float) -> bool:
        """Approximate contact from a sustained position plateau.

        This is deliberately published as an estimate, not bilateral contact.
        It is only used while a close command is active and while the finger
        remains away from the commanded mechanical limit.
        """
        if not bool(self.grasp_control.get("gripper_commanded_closed", False)):
            return False
        cutoff = now - self.gripper_obstruction_window
        samples = [
            (stamp, position)
            for stamp, position in self.gripper_history
            if stamp >= cutoff
        ]
        if len(samples) < 3:
            return False
        if samples[-1][0] - samples[0][0] < self.gripper_obstruction_window * 0.8:
            return False
        positions = [position for _, position in samples]
        away_from_limit = (
            abs(positions[-1] - self.gripper_closed_target)
            > self.gripper_mechanical_limit_tolerance
        )
        return (
            away_from_limit
            and max(positions) - min(positions)
            <= self.gripper_obstruction_max_motion
        )

    def _evaluate(self) -> None:
        now = self._now()
        if now + 1.0e-6 < self.last_clock:
            self._reset_state(clear_baselines=True)
        self.last_clock = now

        obj = self.object_poses.get(self.target_object)
        gripper = self._gripper_pose(obj.stamp if obj is not None else None)
        ready = obj is not None and gripper is not None and self.gripper_position is not None
        if not ready:
            self._publish_outputs(
                False,
                "WAITING_FOR_STATE",
                "",
                {
                    "target_object": self.target_object,
                    "failure_reason": self.failure_reason,
                },
            )
            return

        gripper_commanded_closed = bool(
            self.grasp_control.get("gripper_commanded_closed", False)
        )
        gripper_position_past_threshold = (
            self.gripper_position >= self.closed_threshold
        )
        released = self.gripper_position <= self.released_threshold
        baseline = self.baseline_z.get(self.target_object, obj.position[2])
        center_offset = rotate_vector(
            obj.orientation,
            (
                0.0,
                0.0,
                float(
                    GRASP_CONFIG["object_center_z_offsets_m"][
                        self.target_object
                    ]
                ),
            ),
        )
        center_sample = PoseSample(
            obj.stamp,
            tuple(
                position + offset
                for position, offset in zip(obj.position, center_offset)
            ),
            obj.orientation,
            obj.speed,
        )
        relative_position, relative_orientation = self._relative_pose(
            gripper, center_sample
        )
        distance = vector_norm(relative_position)
        # Retained only as a candidate-search/debug metric.
        near_gripper = distance <= self.capture_radius
        object_profile = OBJECT_GRASP_PROFILES.get(self.target_object, {})
        evaluation_envelope = object_profile.get(
            "loaded_envelope", GRASP_ENVELOPE
        )
        inside_envelope = inside_grasp_envelope(
            relative_position, evaluation_envelope
        )
        frozen = bool(self.scene_grasp.get("frozen", False))
        contact_report_available = bool(
            self.scene_grasp.get("contact_report_available", False)
        )
        bilateral_contact = target_bilateral_contact(
            self.scene_grasp, self.target_object
        )
        geometry_contact_valid = object_loaded_grasp_is_valid(
            self.scene_grasp, self.target_object
        )
        bilateral_contact_frames = int(
            self.scene_grasp.get(
                "bilateral_contact_consecutive_frames", 0
            )
        )
        finger_touching_table = bool(
            self.scene_grasp.get("left_finger_touching_table", False)
            or self.scene_grasp.get(
                "right_finger_touching_table", False
            )
        )
        if (
            contact_report_available
            and "object_touching_table" in self.scene_grasp
        ):
            object_touching_table = bool(
                self.scene_grasp["object_touching_table"]
            )
            object_touching_table_estimated = False
        else:
            object_touching_table = obj.position[2] <= (
                baseline + self.table_clearance
            )
            object_touching_table_estimated = True
        # "Lifted" is a physical clearance label, not an arbitrary 20 cm
        # endpoint.  The assistant separately proves that the configured lift
        # motion completed while the target stayed loaded between both jaws.
        minimum_proof_rise = float(
            object_profile.get("minimum_proof_rise_m", self.table_clearance)
        )
        minimum_table_clearance = float(
            object_profile.get(
                "minimum_table_clearance_m", self.table_clearance
            )
        )
        object_rise = obj.position[2] - baseline
        lifted = (
            not object_touching_table
            and object_rise >= minimum_proof_rise
            and obj.position[2] > baseline + minimum_table_clearance
        )
        if bilateral_contact:
            self.last_target_bilateral_contact_time = now
        bilateral_contact_recent = (
            self.last_target_bilateral_contact_time is not None
            and now - self.last_target_bilateral_contact_time
            <= self.contact_loss_grace
        )
        gripper_obstructed_estimated = self._gripper_obstructed_estimate(now)
        gripper_obstructed = bool(
            self.grasp_control.get("gripper_obstructed", False)
        ) or gripper_obstructed_estimated
        physical_test_lift_active = bool(
            self.grasp_control.get("physical_test_lift_active", False)
        )

        track_relative_pose = (
            gripper_obstructed
            and (inside_envelope or bilateral_contact)
            and not frozen
            and (
                physical_test_lift_active
                or self.physical_grasp_ok
            )
        )
        if track_relative_pose:
            self.relative_history.append((now, relative_position, relative_orientation))
        else:
            self.relative_history.clear()
        stable_relative_pose = self._relative_pose_stable(
            now,
            duration=object_profile.get("stable_duration_s"),
            position_tolerance=object_profile.get(
                "relative_position_tolerance_m"
            ),
            rotation_tolerance=(
                math.radians(
                    float(object_profile["relative_rotation_tolerance_deg"])
                )
                if "relative_rotation_tolerance_deg" in object_profile
                else None
            ),
        )
        contact_angle_range = object_profile.get(
            "gripper_contact_angle_range_rad"
        )
        contact_angle_valid = (
            True
            if not contact_angle_range
            else float(contact_angle_range[0])
            <= self.gripper_position
            <= float(contact_angle_range[1])
        )
        object_up_axis = rotate_vector(obj.orientation, (0.0, 0.0, 1.0))
        cylinder_tilt = math.acos(
            max(-1.0, min(1.0, abs(float(object_up_axis[2]))))
        )
        cylinder_tilt_valid = (
            object_profile.get("strategy") != "cylinder_side_pinch"
            or cylinder_tilt
            <= math.radians(
                float(
                    object_profile.get(
                        "relative_rotation_tolerance_deg", 5.0
                    )
                )
            )
        )
        physical_grasp_candidate = physical_grasp_predicate(
            gripper_obstructed=(gripper_obstructed and contact_angle_valid),
            inside_envelope=inside_envelope,
            object_clear_of_table=lifted,
            stable_relative_pose=stable_relative_pose,
            object_touching_table=object_touching_table,
            finger_touching_table=finger_touching_table,
            bilateral_contact=(geometry_contact_valid and cylinder_tilt_valid),
            contact_report_available=contact_report_available,
        )
        release_started = bool(
            self.grasp_control.get("release_started", False)
        )
        assistant_physical_grasp_verified = bool(
            self.grasp_control.get("physical_grasp_ok", False)
        )
        assistant_verified_candidate = (
            assistant_physical_grasp_verified
            and gripper_obstructed
            and contact_angle_valid
            and cylinder_tilt_valid
            and lifted
            and not object_touching_table
            and not finger_touching_table
            and (
                not contact_report_available
                or geometry_contact_valid
            )
        )
        if (
            physical_test_lift_active and physical_grasp_candidate
        ) or assistant_verified_candidate:
            # The assistant only publishes this acknowledgement after its
            # stricter consecutive direct-PhysX top-lift verification.  Using
            # it closes the asynchronous handoff gap where this evaluator can
            # miss the final candidate frame as lift_active is cleared.
            self.physical_grasp_ok = True
            self.was_grasped = True
            self.failure_reason = ""
        elif (
            self.physical_grasp_ok
            and gripper_commanded_closed
            and contact_report_available
            and not bilateral_contact_recent
            and not release_started
        ):
            # A successful sample is not permanent evidence.  If either real
            # finger loses contact while the grasp command remains active,
            # revoke the latch so a dropped/sliding object cannot be saved as
            # a successful physical grasp.
            self.physical_grasp_ok = False
        grasped = self.physical_grasp_ok and (
            (
                contact_report_available
                and bilateral_contact_recent
            )
            or physical_grasp_candidate
        )
        release_sequence_complete = bool(
            self.grasp_control.get("release_sequence_complete", False)
        )
        if self.was_grasped and released and not self._inside_success_volume(obj.position):
            self.failure_reason = "released_outside_target_box"
        elif (
            self.was_grasped
            and not self._inside_success_volume(obj.position)
            and obj.position[2] <= baseline + 0.01
        ):
            self.failure_reason = "object_dropped"

        inside_box = self._inside_success_volume(obj.position)
        placement_candidate = physical_placement_predicate(
            physical_grasp_was_verified=self.was_grasped,
            release_sequence_complete=release_sequence_complete,
            released=released,
            inside_box=inside_box,
            object_stationary=(
                obj.speed <= self.placement_speed_threshold
            ),
            failure_reason=self.failure_reason,
        )
        if placement_candidate:
            if self.placement_since is None:
                self.placement_since = now
            elif now - self.placement_since >= self.placement_stable_duration:
                self.task_success = True
        else:
            self.placement_since = None

        if self.task_success:
            phase = "SUCCESS"
        elif placement_candidate:
            phase = "PLACING"
        elif self.physical_grasp_ok and gripper_commanded_closed:
            phase = "TRANSPORT"
        elif physical_test_lift_active:
            phase = "PHYSICAL_TEST_LIFT"
        elif gripper_commanded_closed:
            phase = "GRASPING"
        elif lifted:
            phase = "LIFTING"
        else:
            phase = "READY"

        details = {
            "target_object": self.target_object,
            "gripper_position_rad": round(self.gripper_position, 6),
            "closed": gripper_position_past_threshold,
            "released": released,
            "gripper_commanded_closed": gripper_commanded_closed,
            "gripper_position_past_threshold": gripper_position_past_threshold,
            "gripper_obstructed": gripper_obstructed,
            "gripper_obstructed_estimated": gripper_obstructed_estimated,
            "bilateral_contact": bilateral_contact,
            "bilateral_contact_recent": bilateral_contact_recent,
            "reported_bilateral_contact": bool(
                self.scene_grasp.get("bilateral_contact", False)
            ),
            "bilateral_contact_consecutive_frames": (
                bilateral_contact_frames
            ),
            "left_finger_contact_object": self.scene_grasp.get(
                "left_finger_contact_object"
            ),
            "right_finger_contact_object": self.scene_grasp.get(
                "right_finger_contact_object"
            ),
            "left_finger_contact_impulse_ns": self.scene_grasp.get(
                "left_finger_contact_impulse_ns", 0.0
            ),
            "right_finger_contact_impulse_ns": self.scene_grasp.get(
                "right_finger_contact_impulse_ns", 0.0
            ),
            "left_finger_touching_table": bool(
                self.scene_grasp.get(
                    "left_finger_touching_table", False
                )
            ),
            "right_finger_touching_table": bool(
                self.scene_grasp.get(
                    "right_finger_touching_table", False
                )
            ),
            "left_finger_table_impulse_ns": self.scene_grasp.get(
                "left_finger_table_impulse_ns", 0.0
            ),
            "right_finger_table_impulse_ns": self.scene_grasp.get(
                "right_finger_table_impulse_ns", 0.0
            ),
            "contact_report_available": contact_report_available,
            "object_position_m": [round(value, 6) for value in obj.position],
            "object_speed_m_s": round(obj.speed, 6),
            "baseline_z_m": round(baseline, 6),
            "lifted": lifted,
            "gripper_object_distance_m": round(distance, 6),
            "near_gripper": near_gripper,
            "capture_radius_is_debug_only": True,
            "relative_position_gripper_m": [
                round(value, 6) for value in relative_position
            ],
            "inside_grasp_envelope": inside_envelope,
            "grasp_envelope": GRASP_ENVELOPE,
            "relative_pose_stable": stable_relative_pose,
            "object_touching_table": object_touching_table,
            "object_touching_table_estimated": (
                object_touching_table_estimated
            ),
            "frozen": frozen,
            "physical_test_lift_active": physical_test_lift_active,
            "physical_grasp_candidate": physical_grasp_candidate,
            "physical_grasp_ok": self.physical_grasp_ok,
            "release_started": release_started,
            "release_sequence_complete": release_sequence_complete,
            "inside_success_volume": inside_box,
            "placement_stable": self.task_success,
            "failure_reason": self.failure_reason,
        }
        self._publish_outputs(grasped, phase, self.target_object if grasped else "", details)

    def _publish_outputs(
        self, grasped: bool, phase: str, grasped_object: str, details: dict
    ) -> None:
        grasp_message = Bool()
        grasp_message.data = grasped
        self.grasp_pub.publish(grasp_message)
        physical_message = Bool()
        physical_message.data = self.physical_grasp_ok
        self.physical_grasp_pub.publish(physical_message)
        task_message = Bool()
        task_message.data = self.task_success
        self.task_pub.publish(task_message)
        phase_message = String()
        phase_message.data = phase
        self.phase_pub.publish(phase_message)
        object_message = String()
        object_message.data = grasped_object
        self.object_pub.publish(object_message)
        status_message = String()
        status_message.data = json.dumps(
            {
                "phase": phase,
                "grasp_success": grasped,
                "task_success": self.task_success,
                **details,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        self.status_pub.publish(status_message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspEvaluator()
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
