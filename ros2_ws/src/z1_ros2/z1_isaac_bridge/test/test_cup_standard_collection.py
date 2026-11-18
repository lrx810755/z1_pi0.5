"""Cup dimensions and force/lift settings must survive bottle-flow reuse."""
import subprocess
from pathlib import Path
import pytest
from z1_isaac_bridge.grasp_assistant import GRASP_PROFILES
from z1_isaac_bridge.grasp_config import OBJECT_GRASP_PROFILES

@pytest.mark.parametrize('key', [
    'transport_object_z_m', 'physical_test_lift_safe_completion_z_m',
    'physical_test_lift_completion_tolerance_m', 'single_contact_preload_rad',
])
def test_cup_uses_its_own_physical_settings(key):
    assert GRASP_PROFILES['cup'][key] == pytest.approx(OBJECT_GRASP_PROFILES['cup'][key])


def test_cup_alignment_goal_matches_its_reference():
    p=GRASP_PROFILES['cup']
    assert tuple(p['desired_center_'+a+'_m'] for a in 'xyz') == pytest.approx(
        p['calibrated_grasp_reference']['object_in_gripper_local_m'])


def test_cup_launcher_previews_the_bottle_160_episode_cells():
    p=Path('/home/lp531/start_cup_5x4x8_collection.sh')
    assert p.exists(), 'missing cup entrypoint'
    r=subprocess.run(['bash',str(p),'--dry-run'],capture_output=True,text=True,timeout=10)
    assert r.returncode==0,r.stderr
    assert 'target_object:=cup' in r.stdout
    assert 'Z1_COLLECTION_GRID_ROWS=4' in r.stdout
    assert 'Z1_COLLECTION_GRID_COLUMNS=5' in r.stdout
    assert 'external_camera_layout:=dual' in r.stdout
    assert 'z1_tabletop_raw_cup_standard_three_camera_20261019' in r.stdout


def test_cup_height_parameter_moves_tool_without_reusing_bottle_height():
    import copy
    from z1_isaac_bridge import grasp_assistant as m
    physical=copy.deepcopy(OBJECT_GRASP_PROFILES['cup'])
    physical['calibrated_grasp_reference']['reference_grasp_height_m']=0.040
    physical['grasp_height_target_m']=0.040
    baseline=m._cylinder_reference_at_grasp_height(physical)
    physical['grasp_height_target_m']=0.034
    lowered=m._cylinder_reference_at_grasp_height(physical)
    # center minus stator increases by 6 mm when the stator moves down 6 mm.
    key='captured_object_center_minus_stator_world_z_m'
    assert lowered[key]-baseline[key] == pytest.approx(0.006)
    assert physical['calibrated_grasp_reference'][key] == baseline[key]
