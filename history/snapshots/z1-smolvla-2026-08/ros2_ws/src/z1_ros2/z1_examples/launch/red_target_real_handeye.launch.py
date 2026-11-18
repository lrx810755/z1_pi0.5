from pathlib import Path

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _flag(name, enabled):
    return [name] if enabled else []


def launch_setup(context, *args, **kwargs):
    execute = LaunchConfiguration("execute").perform(context).lower() == "true"
    detect_only = LaunchConfiguration("detect_only").perform(context).lower() == "true"
    no_preview = LaunchConfiguration("no_preview").perform(context).lower() == "true"
    wait_for_trigger = LaunchConfiguration("wait_for_trigger").perform(context).lower() == "true"

    z1_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(get_package_share_path("z1_moveit") / "launch" / "z1_moveit.launch.py")
        ),
        launch_arguments={
            "sim_ignition": "false",
            "rviz": LaunchConfiguration("rviz"),
            "starting_controller": "joint_trajectory_controller",
        }.items(),
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(get_package_share_path("realsense2_camera") / "launch" / "rs_launch.py")
        ),
        launch_arguments={
            "camera_namespace": "camera",
            "camera_name": "camera",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "align_depth.enable": LaunchConfiguration("align_depth"),
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_camera")),
    )

    image_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        output="screen",
        arguments=[LaunchConfiguration("image_topic")],
        condition=IfCondition(LaunchConfiguration("image_view")),
    )

    red_target_args = [
        "--calibration",
        LaunchConfiguration("calibration"),
        "--image-topic",
        LaunchConfiguration("image_topic"),
        "--depth-topic",
        LaunchConfiguration("depth_topic"),
        "--camera-info-topic",
        LaunchConfiguration("camera_info_topic"),
        "--base-frame",
        LaunchConfiguration("base_frame"),
        "--gripper-frame",
        LaunchConfiguration("gripper_frame"),
        "--end-effector-link",
        LaunchConfiguration("end_effector_link"),
        "--mode",
        LaunchConfiguration("mode"),
        "--target-distance",
        LaunchConfiguration("target_distance"),
        "--step-size",
        LaunchConfiguration("step_size"),
        "--servo-step-size",
        LaunchConfiguration("servo_step_size"),
        "--servo-gain",
        LaunchConfiguration("servo_gain"),
        "--servo-deadband-px",
        LaunchConfiguration("servo_deadband_px"),
        "--servo-forward-step",
        LaunchConfiguration("servo_forward_step"),
        "--goal-tolerance",
        LaunchConfiguration("goal_tolerance"),
        "--orientation-tolerance",
        LaunchConfiguration("orientation_tolerance"),
        "--orientation-weight",
        LaunchConfiguration("orientation_weight"),
        "--max-joint-delta",
        LaunchConfiguration("max_joint_delta"),
        "--speed-scale",
        LaunchConfiguration("speed_scale"),
        "--detect-timeout",
        LaunchConfiguration("detect_timeout"),
        "--wait-timeout",
        LaunchConfiguration("wait_timeout"),
        "--trigger-service",
        LaunchConfiguration("trigger_service"),
        "--detector",
        LaunchConfiguration("detector"),
        "--yolo-model",
        LaunchConfiguration("yolo_model"),
        "--yolo-class",
        LaunchConfiguration("yolo_class"),
        "--yolo-conf",
        LaunchConfiguration("yolo_conf"),
        "--yolo-device",
        LaunchConfiguration("yolo_device"),
    ]
    red_target_args += _flag("--execute", execute)
    red_target_args += _flag("--detect-only", detect_only)
    red_target_args += _flag("--no-preview", no_preview)
    red_target_args += _flag("--wait-for-trigger", wait_for_trigger)

    red_target_node = Node(
        package="z1_examples",
        executable="red_target_handeye_approach.py",
        output="screen",
        arguments=red_target_args,
    )

    return [z1_moveit_launch, realsense_launch, image_view, red_target_node]


def generate_launch_description():
    calibration_default = str(Path.home() / "ros2_ws" / "z1_handeye_charuco_gripper_to_camera.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("calibration", default_value=calibration_default),
            DeclareLaunchArgument("execute", default_value="false"),
            DeclareLaunchArgument("detect_only", default_value="false"),
            DeclareLaunchArgument("wait_for_trigger", default_value="true"),
            DeclareLaunchArgument("trigger_service", default_value="/red_target/start"),
            DeclareLaunchArgument("detector", default_value="pixel"),
            DeclareLaunchArgument("yolo_model", default_value="yolo11n.pt"),
            DeclareLaunchArgument("yolo_class", default_value="sports ball"),
            DeclareLaunchArgument("yolo_conf", default_value="0.35"),
            DeclareLaunchArgument("yolo_device", default_value=""),
            DeclareLaunchArgument("no_preview", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("image_view", default_value="true"),
            DeclareLaunchArgument("align_depth", default_value="true"),
            DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/camera/aligned_depth_to_color/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera/color/camera_info"),
            DeclareLaunchArgument("base_frame", default_value="link00"),
            DeclareLaunchArgument("gripper_frame", default_value="gripperStator"),
            DeclareLaunchArgument("end_effector_link", default_value="gripperStator"),
            DeclareLaunchArgument("mode", default_value="image-servo"),
            DeclareLaunchArgument("target_distance", default_value="0.45"),
            DeclareLaunchArgument("step_size", default_value="0.05"),
            DeclareLaunchArgument("servo_step_size", default_value="0.005"),
            DeclareLaunchArgument("servo_gain", default_value="0.02"),
            DeclareLaunchArgument("servo_deadband_px", default_value="35.0"),
            DeclareLaunchArgument("servo_forward_step", default_value="0.0"),
            DeclareLaunchArgument("goal_tolerance", default_value="0.008"),
            DeclareLaunchArgument("orientation_tolerance", default_value="0.08"),
            DeclareLaunchArgument("orientation_weight", default_value="1.0"),
            DeclareLaunchArgument("max_joint_delta", default_value="0.12"),
            DeclareLaunchArgument("speed_scale", default_value="0.05"),
            DeclareLaunchArgument("detect_timeout", default_value="10.0"),
            DeclareLaunchArgument("wait_timeout", default_value="20.0"),
            DeclareLaunchArgument("color_profile", default_value="640,480,30"),
            DeclareLaunchArgument("depth_profile", default_value="640,480,30"),
            OpaqueFunction(function=launch_setup),
        ]
    )
