from pathlib import Path

from z1_isaac_bridge import paths


REPOSITORY = Path(__file__).resolve().parents[2]


def test_servo_falls_back_to_installed_user_binary(monkeypatch, tmp_path):
    monkeypatch.delenv('Z1_SERVO_PREFIX', raising=False)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda path: False if str(path).startswith('/opt/ros/') else is_file(path))
    binary = tmp_path / '.local/opt/ros-humble-moveit-servo/opt/ros/humble/lib/moveit_servo/servo_node_main'
    binary.parent.mkdir(parents=True)
    binary.touch()
    assert paths.servo_prefix() / 'lib/moveit_servo/servo_node_main' == binary


def test_environment_overrides_are_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("Z1_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("Z1_ASSET_ROOT", str(tmp_path / "assets"))
    monkeypatch.setenv("ISAAC_SIM_ROOT", str(tmp_path / "isaac"))
    monkeypatch.setenv("Z1_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("Z1_SERVO_PREFIX", str(tmp_path / "servo"))

    assert paths.repository_root() == tmp_path / "repo"
    assert paths.asset_root() == tmp_path / "assets"
    assert paths.isaac_sim_root() == tmp_path / "isaac"
    assert paths.dataset_root("episodes") == tmp_path / "data" / "episodes"
    assert paths.servo_prefix() == tmp_path / "servo"


def test_checkout_asset_fallback_points_inside_repository(monkeypatch):
    monkeypatch.delenv("Z1_REPO_ROOT", raising=False)
    monkeypatch.delenv("Z1_ASSET_ROOT", raising=False)

    assert paths.repository_root() == REPOSITORY
    assert paths.asset_root() == REPOSITORY / "isaac_assets" / "z1"


def test_absolute_dataset_name_does_not_escape_root(monkeypatch, tmp_path):
    monkeypatch.setenv("Z1_DATASET_ROOT", str(tmp_path / "data"))

    assert paths.dataset_root("/episodes") == tmp_path / "data" / "episodes"


def test_symlink_install_finds_repository_in_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    checkout = workspace / "src" / "renamed-checkout"
    (checkout / "z1_isaac_bridge").mkdir(parents=True)
    (checkout / "isaac_assets" / "z1").mkdir(parents=True)
    installed_module = (
        workspace
        / "build"
        / "z1_isaac_bridge"
        / "z1_isaac_bridge"
        / "paths.py"
    )
    monkeypatch.setattr(paths, "__file__", str(installed_module))
    monkeypatch.delenv("Z1_REPO_ROOT", raising=False)

    assert paths.repository_root() == checkout
