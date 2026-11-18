from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[2]
ASSETS = REPOSITORY / "isaac_assets" / "z1"


def test_active_asset_bundle_is_complete():
    required = (
        "generated_dual_camera/smolvla_dual_wrist_camera.usd",
        "generated_dual_camera/z1_with_dual_d435.usd",
        "scripts/run_z1_ros2_scene.py",
        "scripts/layout_sweep.py",
        "scanned_objects/source/apple.usdz",
        "scanned_objects/source/beverage_bottle.usdz",
        "scanned_objects/source/cup.usdz",
        "scanned_objects/source/mouse.usdz",
        "scanned_objects/source/peach.usdz",
        "scanned_objects/ATTRIBUTION.md",
        "scanned_objects/UPSTREAM_CREDITS.csv",
        "scanned_objects/manifest.json",
    )
    missing = [name for name in required if not (ASSETS / name).is_file()]
    assert not missing, missing


def test_asset_bundle_contains_no_machine_artifacts():
    forbidden = {".git", ".codegraph", "__pycache__", "codegraph"}
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "isaac_assets/z1"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    tracked_paths = [REPOSITORY / name for name in tracked if name]
    offenders = [
        path
        for path in tracked_paths
        if forbidden.intersection(path.relative_to(REPOSITORY).parts)
    ]
    offenders += [
        path
        for path in tracked_paths
        if (
            "backup" in path.name.lower()
            or "before_" in path.name
            or path.suffix == ".bak"
        )
    ]
    assert not offenders, offenders
