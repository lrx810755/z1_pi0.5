from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPOSITORY / "isaac_assets" / "z1"


def test_active_runtime_files_do_not_contain_machine_home_path():
    active_files = list((ASSET_ROOT / "scripts").rglob("*.py"))
    active_files += list((ASSET_ROOT / "generated").glob("*.json"))
    active_files += list(
        (ASSET_ROOT / "generated_dual_camera").glob("*.json")
    )
    active_files += list((REPOSITORY / "z1_isaac_bridge" / "launch").glob("*.py"))

    offenders = [
        path.relative_to(REPOSITORY)
        for path in active_files
        if "/home/lp531" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, offenders
