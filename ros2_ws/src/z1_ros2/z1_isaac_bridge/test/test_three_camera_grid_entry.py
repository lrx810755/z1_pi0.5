"""Exercise the user's original entry without starting ROS or collecting data."""
from pathlib import Path
import subprocess
import pytest

ENTRY=Path('/home/lp531/start_smolvla_left_wrist_collection.sh')
LAUNCH=Path('/home/lp531/start_smolvla_balanced_v2_collection.sh')


def dry_run(*args):
    # Fail safely against the old launcher which treats this flag as a quota.
    assert '--dry-run' in LAUNCH.read_text(), 'launcher does not support safe preview'
    return subprocess.run(['bash',str(ENTRY),'--dry-run',*args],text=True,capture_output=True,timeout=10)


def test_original_grid_entry_uses_three_cameras_and_new_dataset():
    r=dry_run('2','5')
    assert r.returncode==0,r.stderr
    assert 'external_camera_layout:=dual' in r.stdout
    assert 'wrist_rgb_topic:=/left_wrist_camera/color/image_raw' in r.stdout
    assert 'vla_wrist_rgb_topic:=none' in r.stdout
    assert 'recording_sync_tolerance_s:=0.045' in r.stdout
    assert 'recording_sync_delay_s:=0.5' in r.stdout
    assert 'freeze_target_keep_collisions:=true' in r.stdout
    assert 'z1_tabletop_raw_left_wrist_three_camera_v1' in r.stdout
    assert 'collect_smolvla_balanced_v2.sh 2 5 ' in r.stdout


@pytest.mark.parametrize('target', ['apple', 'beverage_bottle', 'peach', 'cup'])
def test_selected_object_reaches_scene_and_grid(target):
    r=dry_run('2','5',target)
    assert r.returncode==0,r.stderr
    assert f'target_object:={target}' in r.stdout
    assert r.stdout.splitlines()[-1].strip().endswith(f'progress.json {target}')


def test_invalid_object_is_rejected_before_launch():
    r=dry_run('2','5','unknown')
    assert r.returncode==2
    assert 'ros2 launch' not in r.stdout


@pytest.mark.parametrize('args',[('0','5'),('2','0'),('abc','5')])
def test_bad_quota_or_failure_limit_fails_before_launch(args):
    r=dry_run(*args)
    assert r.returncode==2
    assert 'ros2 launch' not in r.stdout


def test_preview_has_no_directory_side_effects(tmp_path):
    # Explicit source/log overrides preserve the generic entry's original API.
    assert '--dry-run' in LAUNCH.read_text()
    raw=tmp_path/'raw';logs=tmp_path/'logs'
    r=subprocess.run(['bash',str(LAUNCH),'--dry-run','1','3',str(raw),str(logs)],text=True,capture_output=True,timeout=10)
    assert r.returncode==0,r.stderr
    assert not raw.exists() and not logs.exists()


def test_old_two_camera_dataset_is_rejected_before_scene_start(tmp_path):
    import json
    assert '--dry-run' in LAUNCH.read_text()
    raw=tmp_path/'old';raw.mkdir()
    (raw/'dataset_info.json').write_text(json.dumps({'image_shapes':{'wrist':[360,640,3],'global':[480,640,3]}}))
    r=subprocess.run(['bash',str(LAUNCH),'--dry-run','1','3',str(raw),str(tmp_path/'logs')],text=True,capture_output=True,timeout=10)
    assert r.returncode!=0
    assert '三相机' in r.stderr
    assert 'ros2 launch' not in r.stdout


def run_completed_grid(tmp_path, mismatch=False, target='all'):
    import json,os
    raw=tmp_path/'raw';raw.mkdir()
    (raw/'dataset_info.json').write_text(json.dumps({'image_shapes':{
        'wrist':[360,640,3],'global':[480,640,3],'global_right':[480,640,3]}}))
    env=dict(os.environ,TEST_GRID_ROOT=str(raw),TEST_RECORDER_ROOT=str(tmp_path/'wrong') if mismatch else str(raw), TEST_TARGET=target)
    # Export a shell function so sourcing ROS cannot accidentally invoke a real
    # graph. Every quota is already met: no Python collection process runs.
    command='''
ros2() {
  if [[ "$1 $2" == 'param get' ]]; then
    case "$4" in
      dataset_root) printf '%s\\n' "$TEST_RECORDER_ROOT" ;;
      global_right_rgb_topic) echo /global_right_camera/color/image_raw ;;
      *) return 99 ;;
    esac
  elif [[ "$1 $2 $3" == 'run z1_isaac_bridge collection_progress' ]]; then
    echo 2
  else
    echo "unexpected ROS operation" >&2; return 99
  fi
}
export -f ros2
exec bash /home/lp531/ros2_ws/src/z1_ros2/scripts/collect_smolvla_balanced_v2.sh 2 5 "$TEST_GRID_ROOT" "$TEST_GRID_ROOT/progress.json" "$TEST_TARGET"
'''
    return subprocess.run(['bash','-c',command],env=env,text=True,capture_output=True,timeout=15)


def test_completed_grid_resumes_without_recording_and_keeps_all_bottle_cells(tmp_path):
    r=run_completed_grid(tmp_path)
    assert r.returncode==0,r.stderr
    assert '采集完成' in r.stdout
    assert r.stdout.count('已有 2/2 条已保存，跳过')==90
    for row in range(5):
        for col in range(5):
            assert f'方格 ({col},{row}) beverage_bottle：已有 2/2' in r.stdout


def test_grid_rejects_another_recording_root_before_counting_quota(tmp_path):
    r=run_completed_grid(tmp_path,mismatch=True)
    assert r.returncode!=0
    assert '拒绝开始5×5采集' in r.stderr
    assert '已有 2/2' not in r.stdout


def test_bottle_selection_only_visits_25_bottle_cells(tmp_path):
    r=run_completed_grid(tmp_path,target='beverage_bottle')
    assert r.returncode==0,r.stderr
    assert r.stdout.count('已有 2/2 条已保存，跳过')==25
    assert all(target not in r.stdout for target in ('apple', 'peach', 'cup'))
    for row in range(5):
        for col in range(5):
            assert f'方格 ({col},{row}) beverage_bottle：已有 2/2' in r.stdout
