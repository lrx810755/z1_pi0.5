"""Check the collection entry without starting ROS or moving the robot."""
import subprocess
import json
import os


ENTRY = '/home/lp531/start_smolvla_bottle_close_prompt_collection.sh'


def test_collection_preview_selects_separate_data_and_collision_preserving_scene(tmp_path):
    env = dict(os.environ, BOTTLE_CLOSE_PROMPT_DATA_ROOT=str(tmp_path / 'z1_tabletop_raw_bottle_close_prompt_baseline_v1'))
    result = subprocess.run(['bash', ENTRY, '--dry-run'], text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'bottle_recovery_variant:=baseline' in result.stdout
    assert 'freeze_target_keep_collisions:=true' in result.stdout
    assert 'z1_tabletop_raw_bottle_close_prompt_baseline_v1' in result.stdout
    assert '--episodes 30' in result.stdout
    assert '--max-attempts 150' in result.stdout
    result = subprocess.run(['bash', ENTRY, '--dry-run', 'left', '6', '30'],
                            text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'bottle_recovery_variant:=left' in result.stdout
    assert '--episodes 6' in result.stdout


def test_bad_collection_arguments_rejected_without_launching_scene():
    for args in [('unknown',), ('baseline', '0'), ('baseline', '6', '2')]:
        result = subprocess.run(['bash', ENTRY, *args], text=True, capture_output=True)
        assert result.returncode == 2


def test_resume_collects_only_missing_successful_episodes(tmp_path):
    for number, success in [(0, True), (1, False)]:
        ep = tmp_path / f'episode_{number:06d}'
        ep.mkdir()
        (ep / 'episode_summary.json').write_text(json.dumps({'task_success': success}))
    env = dict(os.environ, BOTTLE_CLOSE_PROMPT_DATA_ROOT=str(tmp_path))
    result = subprocess.run(['bash', ENTRY, '--dry-run', 'baseline', '3', '10'],
                            text=True, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    assert '--episodes 2' in result.stdout
