"""Start the ROS 2 side of the Z1 Isaac Sim integration."""

from pathlib import Path
from ament_index_python.packages import get_package_share_path
import json
import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    SetEnvironmentVariable, OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from z1_isaac_bridge.paths import asset_root, dataset_root, isaac_sim_root


def camera_transforms(context):
    if LaunchConfiguration("external_camera_layout").perform(context) == "dual":
        configs = json.loads((get_package_share_path("z1_isaac_bridge") / "config" / "external_cameras.json").read_text())
    else:
        configs = {"global": {"position_m": [0.72, -0.72, 0.72],
            "orientation_xyzw": [-0.8618642404075918, -0.2180293171642894, 0.11229407851178168, 0.44389558219778247],
            "frame_id": "global_camera_optical_frame"}}
    nodes = []
    for name, cfg in configs.items():
        args = []
        for key, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), (*cfg["position_m"], *cfg["orientation_xyzw"])):
            args.extend(["--" + key, str(value)])
        args.extend(["--frame-id", "world", "--child-frame-id", cfg["frame_id"]])
        nodes.append(Node(package="tf2_ros", executable="static_transform_publisher",
            name=name + "_camera_static_tf", arguments=args, parameters=[{"use_sim_time": True}],
            condition=IfCondition(LaunchConfiguration("simulation_nodes"))))
    return nodes


def generate_launch_description():
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

    assets = asset_root()
    isaac_root = isaac_sim_root()
    scene_default = str(
        assets
        / "generated_dual_camera"
        / "smolvla_dual_wrist_camera.usd"
    )
    script_default = str(assets / "scripts" / "run_z1_ros2_scene.py")
    isaac_python_default = str(isaac_root / "python.sh")
    rviz_config_default = str(moveit_share / "config" / "moveit.rviz")
    grasp_config = '/home/lp531/robot/reverse8_queue_20261112/bottle_corrections/grasp_physics.json'
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
        PythonLaunchDescriptionSource(str(Path(__file__).resolve().parent / "sequence_move_group.launch.py")),
        launch_arguments={"use_sim_time": "true", "capabilities": "pilz_industrial_motion_planner/MoveGroupSequenceService"}.items(),
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
        "--eval-table-friction",
        LaunchConfiguration("eval_table_friction"),
        "--eval-arm-max-velocity",
        LaunchConfiguration("eval_arm_max_velocity"),
        "--external-camera-layout", LaunchConfiguration("external_camera_layout"),
    ]
    isaac_environment = {
        "ROS_DISTRO": "humble",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "LD_LIBRARY_PATH": str(
            isaac_root / "exts" / "isaacsim.ros2.bridge" / "humble" / "lib"
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
    smolvla_rollout = ExecuteProcess(
        cmd=[
            "/home/lp531/miniconda3/envs/unifolm-vla/bin/python",
            "-m",
            "z1_isaac_bridge.smolvla_sim_rollout",
            "--checkpoint",
            LaunchConfiguration("smolvla_checkpoint"),
            "--objects",
            LaunchConfiguration("target_object"),
            "--episodes-per-object",
            LaunchConfiguration("smolvla_episodes"),
            "--timeout",
            LaunchConfiguration("smolvla_episode_timeout"),
            "--output",
            LaunchConfiguration("smolvla_output"),
        ],
        additional_env={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_smolvla_policy")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "fastdds_profile",
                default_value=str(
                    get_package_share_path("z1_isaac_bridge")
                    / "config" / "fastdds_camera.xml"
                ),
                description="Fast DDS transport profile sized for three raw RGB cameras",
            ),
            SetEnvironmentVariable(
                "FASTRTPS_DEFAULT_PROFILES_FILE", LaunchConfiguration("fastdds_profile")
            ),
            SetEnvironmentVariable("FASTDDS_DEFAULT_PROFILES_FILE", LaunchConfiguration("fastdds_profile")),
            DeclareLaunchArgument("external_camera_layout", default_value="legacy", choices=["legacy", "dual"]),
            DeclareLaunchArgument("start_isaac", default_value="true"),
            DeclareLaunchArgument("eval_table_friction", default_value="0"),
            DeclareLaunchArgument("eval_arm_max_velocity", default_value="0"),
            DeclareLaunchArgument("simulation_nodes", default_value="true"),
            DeclareLaunchArgument("moveit", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("target_object", default_value="apple"),
            DeclareLaunchArgument("bottle_recovery_variant", default_value="baseline"),
            DeclareLaunchArgument("freeze_target_keep_collisions", default_value="false"),
            DeclareLaunchArgument("run_smolvla_policy", default_value="false"),
            DeclareLaunchArgument(
                "smolvla_checkpoint",
                default_value=(
                    "/home/lp531/smolvla_training/"
                    "z1_4x100_smolvla_20260921_113511/"
                    "checkpoints/last/pretrained_model"
                ),
            ),
            DeclareLaunchArgument("smolvla_episodes", default_value="1"),
            DeclareLaunchArgument(
                "smolvla_episode_timeout", default_value="60"
            ),
            DeclareLaunchArgument(
                "smolvla_output",
                default_value="/home/lp531/smolvla_eval",
            ),
            DeclareLaunchArgument("task_gui", default_value="true"),
            DeclareLaunchArgument("task_gui_port", default_value="8767"),
            DeclareLaunchArgument(
                "dataset_root",
                default_value=str(dataset_root("z1_tabletop_raw_left_wrist_v1")),
            ),
            DeclareLaunchArgument("recording_fps", default_value="10.0"),
            DeclareLaunchArgument(
                "recording_sync_tolerance_s", default_value="0.045"
            ),
            DeclareLaunchArgument(
                "recording_sync_delay_s", default_value="0.5"
            ),
            DeclareLaunchArgument(
                "recording_action_age_s", default_value="0.40"
            ),
            DeclareLaunchArgument(
                "wrist_rgb_topic",
                default_value="/left_wrist_camera/color/image_raw",
            ),
            DeclareLaunchArgument(
                "vla_wrist_rgb_topic",
                default_value="none",
            ),
            DeclareLaunchArgument("scene", default_value=scene_default),
            DeclareLaunchArgument("isaac_script", default_value=script_default),
            DeclareLaunchArgument("isaac_python", default_value=isaac_python_default),
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
                condition=IfCondition(LaunchConfiguration("simulation_nodes")),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="grasp_assistant",
                output="screen",
                parameters=[
                    {"use_sim_time": True},
                    {"target_object": LaunchConfiguration("target_object")},
                    {"bottle_recovery_variant": LaunchConfiguration("bottle_recovery_variant")},
                    {"freeze_target_keep_collisions": LaunchConfiguration("freeze_target_keep_collisions")},
                ],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            LaunchConfiguration("simulation_nodes"),
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
                        "sync_tolerance_s": ParameterValue(
                            LaunchConfiguration(
                                "recording_sync_tolerance_s"
                            ),
                            value_type=float,
                        )
                    },
                    {
                        "sync_delay_s": ParameterValue(
                            LaunchConfiguration("recording_sync_delay_s"),
                            value_type=float,
                        )
                    },
                    {
                        "action_age_s": ParameterValue(
                            LaunchConfiguration("recording_action_age_s"),
                            value_type=float,
                        )
                    },
                    {
                        "global_right_rgb_topic": ParameterValue(PythonExpression([
                            "'/global_right_camera/color/image_raw' if '", LaunchConfiguration("external_camera_layout"), "' == 'dual' else 'none'"
                        ]), value_type=str),
                        "camera_calibration_path": str(get_package_share_path("z1_isaac_bridge") / "config" / "external_cameras.json"),
                        "wrist_rgb_topic": ParameterValue(
                            LaunchConfiguration("wrist_rgb_topic"),
                            value_type=str,
                        )
                    },
                    {
                        "vla_wrist_rgb_topic": ParameterValue(
                            LaunchConfiguration("vla_wrist_rgb_topic"),
                            value_type=str,
                        )
                    },
                ],
                condition=IfCondition(LaunchConfiguration("simulation_nodes")),
            ),
            Node(
                package="z1_isaac_bridge",
                executable="episode_reset",
                output="screen",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("simulation_nodes")),
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
                            LaunchConfiguration("simulation_nodes"),
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
                condition=IfCondition(LaunchConfiguration("simulation_nodes")),
            ),
            OpaqueFunction(function=camera_transforms),
            move_group,
            rviz,
            isaac_gui,
            isaac_headless,
            smolvla_rollout,
        ]
    )
