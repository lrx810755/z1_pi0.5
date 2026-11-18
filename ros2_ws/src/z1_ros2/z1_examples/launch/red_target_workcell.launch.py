from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
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

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(get_package_share_path("realsense2_camera") / "launch" / "rs_launch.py")
        ),
        launch_arguments={
            "camera_namespace": "camera",
            "camera_name": "camera",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_camera")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("color_profile", default_value="640,480,30"),
            DeclareLaunchArgument("depth_profile", default_value="640,480,30"),
            moveit,
            realsense,
        ]
    )
