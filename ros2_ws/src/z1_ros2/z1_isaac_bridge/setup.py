from glob import glob
import os

from setuptools import find_packages, setup

package_name = "z1_isaac_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            "share/" + package_name,
            [
                "package.xml",
                "README.md",
                "PHYSICAL_GRASP_VALIDATION.md",
                "AGENT_HANDOFF.md",
                "SMOLVLA_SIM_EVAL.md",
            ],
        ),
        (os.path.join("share", package_name, "launch"), glob("launch/*launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.json")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config"), glob("config/*.xml")),
        (os.path.join("share", package_name, "web"), glob("web/*.html")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lp531",
    maintainer_email="lp531@example.com",
    description="ROS 2 trajectory and launch integration for the Unitree Z1 in Isaac Sim.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "grasp_assistant = z1_isaac_bridge.grasp_assistant:main",
            "grasp_evaluator = z1_isaac_bridge.grasp_evaluator:main",
            "grasp_gui_bridge = z1_isaac_bridge.grasp_gui_bridge:main",
            "episode_recorder = z1_isaac_bridge.episode_recorder:main",
            "episode_reset = z1_isaac_bridge.episode_reset:main",
            "object_collision_sync = z1_isaac_bridge.object_collision_sync:main",
            "planning_scene_sync = z1_isaac_bridge.planning_scene_sync:main",
            "trajectory_bridge = z1_isaac_bridge.trajectory_bridge:main",
            "moveit_servo_vendor = z1_isaac_bridge.servo_node_vendor:main",
            "continuous_training = z1_isaac_bridge.continuous_training:main",
            "select_balanced_dataset = z1_isaac_bridge.dataset_selection:main",
            "collection_progress = z1_isaac_bridge.dataset_selection:progress_main",
            "smolvla_sim_rollout = z1_isaac_bridge.smolvla_sim_rollout:main",
            "smolvla_voice_backend = z1_isaac_bridge.smolvla_voice_backend:main",
        ],
    },
)
