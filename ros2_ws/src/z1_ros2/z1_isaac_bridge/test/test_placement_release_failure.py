"""Release above the target must be allowed to settle before placement judgment."""
import pytest
from z1_isaac_bridge.grasp_evaluator import placement_failure_reason, physical_placement_predicate


@pytest.mark.parametrize('changes, expected', [
    ({}, ''),  # Verified bottle falling above the correct box.
    ({'inside_box_footprint': False}, 'released_outside_target_box'),
    ({'released': False, 'release_started': False, 'object_near_table': True}, 'object_dropped'),
    ({'released': False, 'object_near_table': True}, ''),  # Intentional release.
    ({'physical_grasp_was_verified': False, 'inside_box_footprint': False}, ''),
    ({'inside_box': True, 'object_near_table': True}, ''),
])
def test_release_failure_classification(changes, expected):
    fields = dict(physical_grasp_was_verified=True, released=True,
                  release_started=True, inside_box_footprint=True,
                  inside_box=False, object_near_table=False)
    fields.update(changes)
    assert placement_failure_reason(**fields) == expected


@pytest.mark.parametrize('inside_box, stationary, expected', [
    (False, False, False), (False, True, False), (True, False, False), (True, True, True),
])
def test_release_permission_still_requires_actual_stable_placement(inside_box, stationary, expected):
    assert physical_placement_predicate(physical_grasp_was_verified=True,
        release_sequence_complete=True, released=True, inside_box=inside_box,
        object_stationary=stationary, failure_reason='') == expected


@pytest.mark.parametrize('verified, release_complete', [(False, True), (True, False), (False, False)])
def test_stationary_bottle_in_box_requires_grasp_and_complete_release(verified, release_complete):
    assert not physical_placement_predicate(physical_grasp_was_verified=verified,
        release_sequence_complete=release_complete, released=True, inside_box=True,
        object_stationary=True, failure_reason='')
