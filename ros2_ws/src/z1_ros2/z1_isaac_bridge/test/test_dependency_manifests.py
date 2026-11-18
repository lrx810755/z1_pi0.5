from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
FORBIDDEN_PIP_PACKAGES = ("rclpy", "isaacsim", "omni", "pxr", "carb", "usdrt")


def requirement_text(name):
    return (REPOSITORY / name).read_text(encoding="utf-8")


def test_requirements_are_split_by_runtime_owner():
    assert "-r requirements/python.txt" in requirement_text("requirements.txt")
    assert "lerobot==0.3.4" in requirement_text("requirements/training.txt")

    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (REPOSITORY / "requirements").glob("*.txt")
    )
    assert not [name for name in FORBIDDEN_PIP_PACKAGES if name in combined]


def test_ros_package_manifests_own_system_python_dependencies():
    bridge = requirement_text("z1_isaac_bridge/package.xml")
    examples = requirement_text("z1_examples/package.xml")

    assert "python3-yaml" in bridge
    for dependency in (
        "cv_bridge",
        "python3-numpy",
        "python3-opencv",
        "python3-pyqt5",
        "python3-scipy",
        "python3-yaml",
    ):
        assert dependency in examples
