"""Start the ROS 2 side of the Z1 Isaac Sim integration."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_path
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def _discover_repo_root() -> Path:
    configured = os.environ.get("Z1_SMOLVLA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    origin = Path(__file__).resolve()
    for parent in origin.parents:
        if (parent / "isaac_assets" / "z1").is_dir() and (
            parent / "ros2_ws" / "src" / "z1_ros2"
        ).is_dir():
            return parent
    raise RuntimeError(
        "Unable to discover the Z1 SmolVLA repository. "
        "Source scripts/setup_env.sh or set Z1_SMOLVLA_ROOT."
    )


def generate_launch_description():
    repo_root = _discover_repo_root()
    asset_root = Path(
        os.environ.get("Z1_ASSET_ROOT", repo_root / "isaac_assets" / "z1")
    ).expanduser()
    data_root = Path(
        os.environ.get("SMOLVLA_DATA_ROOT", repo_root / "data" / "runtime")
    ).expanduser()
    isaac_sim_root = Path(
        os.environ.get("ISAAC_SIM_ROOT", Path.home() / "isaac-sim")
    ).expanduser()
    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    description_share = get_package_share_path("z1_description")
    moveit_share = get_package_share_path("z1_moveit")
    urdf = description_share / "urdf" / "z1.urdf.xacro"
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                str(urdf),
                " sim_ignition:=true",
                " with_gripper:=true",
            ]
        ),
        value_type=str,
    )

    scene_default = str(
        asset_root / "generated_dual_camera" / "smolvla_dual_wrist_camera.usd"
    )
    script_default = str(asset_root / "scripts" / "run_z1_ros2_scene.py")
    isaac_python_default = str(isaac_sim_root / "python.sh")
    rviz_config_default = str(moveit_share / "config" / "moveit.rviz")
    grasp_config = str(
        get_package_share_path("z1_isaac_bridge")
        / "config"
        / "grasp_physics.json"
    )
    servo_config = (
        get_package_share_path("z1_isaac_bridge")
        / "config"
        / "z1_servo.yaml"
    )
    with servo_config.open("r", encoding="utf-8") as stream:
        servo_parameters = {"moveit_servo": yaml.safe_load(stream)}
    moveit_config = MoveItConfigsBuilder(
        "z1_description", package_name="z1_moveit"
    ).to_moveit_configs()

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(moveit_share / "launch" / "move_group.launch.py")),
        launch_arguments={"use_sim_time": "true"}.items(),
        condition=IfCondition(LaunchConfiguration("moveit")),
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(moveit_share / "launch" / "moveit_rviz.launch.py")),
        launch_arguments={
            "use_sim_time": "true",
            "rviz_config": LaunchConfiguration("rviz_config"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # Isaac Sim ships Python 3.11 while ROS Humble uses Python 3.10. Strip the
    # sourced workspace paths and let Isaac use its bundled Humble bridge.
    isaac_command = [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        "-u",
        "AMENT_PREFIX_PATH",
        "-u",
        "COLCON_PREFIX_PATH",
        "-u",
        "CMAKE_PREFIX_PATH",
        "-u",
        "ROS_DISTRO",
        "-u",
        "ROS_VERSION",
        LaunchConfiguration("isaac_python"),
        LaunchConfiguration("isaac_script"),
        "--scene",
        LaunchConfiguration("scene"),
        "--grasp-config",
        grasp_config,
    ]
    isaac_environment = {
        "ROS_DISTRO": ros_distro,
        "RMW_IMPLEMENTATION": os.environ.get(
            "RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"
        ),
        "ISAAC_SIM_ROOT": LaunchConfiguration("isaac_sim_root"),
        "Z1_SMOLVLA_ROOT": str(repo_root),
        "Z1_ASSET_ROOT": str(asset_root),
        "SMOLVLA_DATA_ROOT": str(data_root),
        "LD_LIBRARY_PATH": PathJoinSubstitution(
            [
                LaunchConfiguration("isaac_sim_root"),
                "exts",
                "isaacsim.ros2.bridge",
                ros_distro,
                "lib",
            ]
        ),
        "Z1_ISAAC_GRASP_CONFIG": grasp_config,
    }
    start_gui = PythonExpression(
        [
            "'",
            LaunchConfiguration("start_isaac"),
            "' == 'true' and '",
            LaunchConfiguration("headless"),
            "' != 'true'",
        ]
    )
    start_headless = PythonExpression(
        [
            "'",
            LaunchConfiguration("start_isaac"),
            "' == 'true' and '",
            LaunchConfiguration("headless"),
            "' == 'true'",
        ]
    )
    isaac_gui = ExecuteProcess(
        cmd=isaac_command,
        additional_env=isaac_environment,
        output="screen",
        condition=IfCondition(start_gui),
    )
    isaac_headless = ExecuteProcess(
        cmd=isaac_command + ["--headless"],
        additional_env={
            **isaac_environment,
        },
        output="screen",
        condition=IfCondition(start_headless),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_isaac", default_value="true"),
            DeclareLaunchArgument("moveit", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("target_object", default_value="apple"),
            DeclareLaunchArgument("task_gui", default_value="true"),
            DeclareLaunchArgument("task_gui_port", default_value="8767"),
            DeclareLaunchArgument(
                "dataset_root",
                default_value=str(data_root / "z1_tabletop_raw_dual_wrist"),
            ),
            DeclareLaunchArgument("recording_fps", default_value="10.0"),
            DeclareLaunchArgument(
                "vla_wrist_rgb_topic",
                default_value="/vla_wrist_camera/color/image_raw",
            ),
            DeclareLaunchArgument("scene", default_value=scene_default),
            DeclareLaunchArgument("isaac_script", default_value=script_default),
            DeclareLaunchArgument("isaac_python", default_value=isaac_python_default),
            DeclareLaunchArgument(
                "isaac_sim_root", default_value=str(isaac_sim_root)
            ),
            DeclareLaunchArgument("rviz_config", default_value=rviz_config_default),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[
                    {"robot_description": robot_description},
                    {"use_sim_time": True},
                ],
            ),
            Node(
                package="z1_isaac_bridge",
                executable="trajectory_bridge",
                output="screen",
                parameters=[{"use_sim_time": True}],
            ),
            Node(
                package="z1_isaac_bridge",
                executable="moveit_servo_vendor",
                name="servo_node",
                output="screen",
                parameters=[
                    servo_parameters,
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    {"use_sim_time": True},
                ],
                condition=IfCondition(LaunchConfiguration("moveit")),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="grasp_evaluator",
                output="screen",
                parameters=[
                    {"use_sim_time": True},
                    {"target_object": LaunchConfiguration("target_object")},
                ],
                condition=IfCondition(LaunchConfiguration("start_isaac")),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="grasp_assistant",
                output="screen",
                parameters=[
                    {"use_sim_time": True},
                    {"target_object": LaunchConfiguration("target_object")},
                ],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            LaunchConfiguration("start_isaac"),
                            "' == 'true' and '",
                            LaunchConfiguration("moveit"),
                            "' == 'true'",
                        ]
                    )
                ),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="grasp_gui_bridge",
                output="screen",
                parameters=[
                    {
                        "port": ParameterValue(
                            LaunchConfiguration("task_gui_port"), value_type=int
                        )
                    }
                ],
                condition=IfCondition(LaunchConfiguration("task_gui")),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="episode_recorder",
                output="screen",
                parameters=[
                    {"use_sim_time": True},
                    {"dataset_root": LaunchConfiguration("dataset_root")},
                    {
                        "fps": ParameterValue(
                            LaunchConfiguration("recording_fps"),
                            value_type=float,
                        )
                    },
                    {
                        "vla_wrist_rgb_topic": ParameterValue(
                            LaunchConfiguration("vla_wrist_rgb_topic"),
                            value_type=str,
                        )
                    },
                ],
                condition=IfCondition(LaunchConfiguration("start_isaac")),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="episode_reset",
                output="screen",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("start_isaac")),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="planning_scene_sync",
                output="screen",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("moveit")),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="object_collision_sync",
                output="screen",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            LaunchConfiguration("start_isaac"),
                            "' == 'true' and '",
                            LaunchConfiguration("moveit"),
                            "' == 'true'",
                        ]
                    )
                ),
            ),
            Node(
                package="z1_examples",
                executable="camera_info_fix.py",
                output="screen",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("start_isaac")),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="global_camera_static_tf",
                arguments=[
                    "--x",
                    "0.72",
                    "--y",
                    "-0.72",
                    "--z",
                    "0.72",
                    "--qx",
                    "-0.8618642404075918",
                    "--qy",
                    "-0.2180293171642894",
                    "--qz",
                    "0.11229407851178168",
                    "--qw",
                    "0.44389558219778247",
                    "--frame-id",
                    "world",
                    "--child-frame-id",
                    "global_camera_optical_frame",
                ],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("start_isaac")),
            ),
            move_group,
            rviz,
            isaac_gui,
            isaac_headless,
        ]
    )
