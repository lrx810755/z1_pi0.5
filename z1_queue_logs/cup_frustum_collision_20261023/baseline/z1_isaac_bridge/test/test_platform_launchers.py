from pathlib import Path
import xml.etree.ElementTree as ET


REPOSITORY = Path(__file__).resolve().parents[2]


def transport_types(path):
    root = ET.parse(path).getroot()
    return {
        element.text
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "type"
    }


def test_native_and_wsl_transport_contracts():
    config = REPOSITORY / "z1_isaac_bridge" / "config"
    assert {"SHM", "UDPv4"} <= transport_types(config / "fastdds_camera.xml")
    assert transport_types(config / "fastdds_wsl2.xml") == {"UDPv4"}


def test_shell_launchers_fail_fast():
    for name in (
        "configure_environment.sh",
        "bootstrap_ubuntu_22_04.sh",
        "start_ros_stack.sh",
    ):
        source = (REPOSITORY / "scripts" / name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in source


def test_windows_launcher_has_balanced_delimiters():
    source = (
        REPOSITORY / "windows" / "start_isaac_sim.ps1"
    ).read_text(encoding="utf-8")
    for opening, closing in (("(", ")"), ("{", "}"), ("[", "]")):
        assert source.count(opening) == source.count(closing)
    assert "python.bat" in source
    assert "run_z1_ros2_scene.py" in source


def test_reproduction_verifier_is_read_only_and_checks_assets():
    source = (
        REPOSITORY / "scripts" / "verify_reproduction.sh"
    ).read_text(encoding="utf-8")
    assert "set -euo pipefail" in source
    assert "git lfs" in source
    assert "sha256" in source
    assert "compileall" in source
    assert "github" + "_pat_" in source


def test_external_isaac_keeps_ros_integration_nodes_enabled():
    launch = (
        REPOSITORY / "z1_isaac_bridge" / "launch" / "z1_isaac.launch.py"
    ).read_text(encoding="utf-8")
    assert 'DeclareLaunchArgument("simulation_nodes", default_value="true")' in launch
    assert 'condition=IfCondition(LaunchConfiguration("start_isaac"))' not in launch


def test_smolvla_voice_launcher_owns_only_processes_it_starts():
    source = Path("/home/lp531/start_smolvla_voice.sh").read_text(
        encoding="utf-8"
    )
    assert "set -euo pipefail" in source
    assert source.index("source /opt/ros/humble/setup.bash") < source.index(
        "set -euo pipefail"
    )
    assert "/smolvla/randomize_episode_full" in source
    assert "scene_started_by_script=0" in source
    assert 'if (( scene_started_by_script )); then' in source
    assert "z1_4x100_state13_2cam_trimmed_chunk30_20260921_225434" in source
    assert "smolvla_voice_backend" in source
    assert "smolvla_voice_frontend.py" in source
    stop = "ros2 topic pub --once /smolvla/voice_command"
    assert stop in source
    assert source.index(stop) < source.index('kill -- "-${backend_pid}"')


def test_balanced_collection_launcher_is_resumable_and_unbuffered():
    collector = (
        REPOSITORY / "scripts" / "collect_smolvla_balanced_v2.sh"
    ).read_text(encoding="utf-8")
    launcher = Path(
        "/home/lp531/start_smolvla_balanced_v2_collection.sh"
    ).read_text(encoding="utf-8")

    assert "collection_progress" in collector
    assert '--episodes "$((quota - existing))"' in collector
    assert "已保存，跳过" in collector
    assert "PYTHONUNBUFFERED=1" in launcher
    assert 'tee -a "$log_dir/collection.log"' in launcher
