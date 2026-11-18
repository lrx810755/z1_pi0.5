from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_path


def generate_launch_description():
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(get_package_share_path("z1_moveit") / "launch" / "z1_moveit.launch.py")
        ),
        launch_arguments={
            "sim_ignition": "false",
            "rviz": "true",
            "starting_controller": "joint_trajectory_controller",
        }.items(),
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(
                get_package_share_path("realsense2_camera")
                / "launch"
                / "rs_launch.py"
            )
        ),
        launch_arguments={
            "camera_namespace": "camera",
            "camera_name": "camera",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "rgb_camera.color_profile": "640,480,30",
            "depth_module.depth_profile": "640,480,30",
        }.items(),
    )

    red_ball_gui = Node(
        package="z1_examples",
        executable="red_target_gui.py",
        output="screen",
    )

    return LaunchDescription([moveit, camera, red_ball_gui])
