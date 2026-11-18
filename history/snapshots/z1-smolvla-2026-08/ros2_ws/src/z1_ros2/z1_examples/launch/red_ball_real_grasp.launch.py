from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_path


def launch_setup(context, *args, **kwargs):
    execute = LaunchConfiguration("execute").perform(context).lower() == "true"
    no_preview = LaunchConfiguration("no_preview").perform(context).lower() == "true"
    red_ball_args = [
        "--image-topic", LaunchConfiguration("image_topic"),
        "--depth-topic", LaunchConfiguration("depth_topic"),
        "--camera-info-topic", LaunchConfiguration("camera_info_topic"),
        "--camera-frame", LaunchConfiguration("camera_frame"),
        "--base-frame", LaunchConfiguration("base_frame"),
        "--end-effector-link", LaunchConfiguration("end_effector_link"),
        "--execution-mode", LaunchConfiguration("execution_mode"),
        "--approach-distance", LaunchConfiguration("approach_distance"),
        "--stop-distance", LaunchConfiguration("stop_distance"),
        "--speed-scale", LaunchConfiguration("speed_scale"),
        "--detect-timeout", LaunchConfiguration("detect_timeout"),
        "--wait-timeout", LaunchConfiguration("wait_timeout"),
        "--depth-u-scale", LaunchConfiguration("depth_u_scale"),
        "--depth-v-scale", LaunchConfiguration("depth_v_scale"),
        "--detector", LaunchConfiguration("detector"),
        "--yolo-model", LaunchConfiguration("yolo_model"),
        "--yolo-class", LaunchConfiguration("yolo_class"),
        "--yolo-conf", LaunchConfiguration("yolo_conf"),
        "--yolo-device", LaunchConfiguration("yolo_device"),
    ]
    red_ball_args.append("--execute" if execute else "--no-execute")
    if no_preview:
        red_ball_args.append("--no-preview")

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(get_package_share_path("z1_moveit") / "launch" / "z1_moveit.launch.py")
        ),
        launch_arguments={
            "sim_ignition": "false",
            "rviz": LaunchConfiguration("rviz"),
            "starting_controller": "joint_trajectory_controller",
        }.items(),
    )
    camera = IncludeLaunchDescription(
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
    )
    image_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        arguments=[LaunchConfiguration("image_topic")],
        condition=IfCondition(LaunchConfiguration("image_view")),
    )
    detector = Node(
        package="z1_examples",
        executable="red_ball_camera_moveit.py",
        output="screen",
        arguments=red_ball_args,
    )
    return [moveit, camera, image_view, detector]


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("execute", default_value="false"),
        DeclareLaunchArgument("no_preview", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("image_view", default_value="true"),
        DeclareLaunchArgument("align_depth", default_value="true"),
        DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera/color/camera_info"),
        DeclareLaunchArgument("camera_frame", default_value="camera_color_optical_frame"),
        DeclareLaunchArgument("base_frame", default_value="link00"),
        DeclareLaunchArgument("end_effector_link", default_value="link06"),
        DeclareLaunchArgument("execution_mode", default_value="moveit"),
        DeclareLaunchArgument("approach_distance", default_value="0.10"),
        DeclareLaunchArgument("stop_distance", default_value="0.12"),
        DeclareLaunchArgument("speed_scale", default_value="0.05"),
        DeclareLaunchArgument("detect_timeout", default_value="10.0"),
        DeclareLaunchArgument("wait_timeout", default_value="20.0"),
        DeclareLaunchArgument("depth_u_scale", default_value="1.0"),
        DeclareLaunchArgument("depth_v_scale", default_value="1.0"),
        DeclareLaunchArgument("detector", default_value="pixel"),
        DeclareLaunchArgument("yolo_model", default_value="yolo11n.pt"),
        DeclareLaunchArgument("yolo_class", default_value="sports ball"),
        DeclareLaunchArgument("yolo_conf", default_value="0.35"),
        DeclareLaunchArgument("yolo_device", default_value=""),
        DeclareLaunchArgument("color_profile", default_value="640,480,30"),
        DeclareLaunchArgument("depth_profile", default_value="640,480,30"),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=launch_setup)])
