from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("z1_description", package_name="z1_moveit").to_moveit_configs()
    moveit_config.trajectory_execution["use_sim_time"] = ParameterValue(
        LaunchConfiguration("use_sim_time"), value_type=bool
    )
    # Isaac may need a short physics-settling interval after the last planned
    # waypoint. Keep MoveIt from canceling the controller before that interval
    # has elapsed.
    moveit_config.trajectory_execution.update(
        {
            "trajectory_execution.allowed_execution_duration_scaling": 2.0,
            "trajectory_execution.allowed_goal_duration_margin": 3.0,
            "trajectory_execution.allowed_start_tolerance": 0.01,
        }
    )
    launch_description = LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ])
    for action in generate_move_group_launch(moveit_config).entities:
        launch_description.add_action(action)
    return launch_description
