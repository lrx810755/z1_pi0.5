import threading
from types import SimpleNamespace
import pytest

from moveit_msgs.msg import AllowedCollisionMatrix
from z1_isaac_bridge.object_collision_sync import ObjectCollisionSync, acm_permissions
from z1_isaac_bridge.grasp_assistant import GraspAssistant


def node_with_old_permissions():
    node = ObjectCollisionSync.__new__(ObjectCollisionSync)
    node._base_acm = AllowedCollisionMatrix()
    node.touch_links = ['link06', 'camera_link', 'gripperStator', 'gripperMover']
    node._bottle_finger_contact = False
    names = ['smolvla_object_beverage_bottle', *node.touch_links, 'link05']
    for name in names:
        node._ensure_acm_name(node._base_acm, name)
    # A restarted synchronizer must clear permissions left by the old node.
    for row in node._base_acm.entry_values:
        row.enabled = [True] * len(names)
    return node


def allowed(matrix, a, b):
    return matrix.entry_values[matrix.entry_names.index(a)].enabled[matrix.entry_names.index(b)]


def test_bottle_approach_clears_legacy_collision_permissions():
    node = node_with_old_permissions()
    matrix = node._target_acm('beverage_bottle')
    for link in [*node.touch_links, 'link05']:
        assert not allowed(matrix, 'smolvla_object_beverage_bottle', link)
        assert not allowed(matrix, link, 'smolvla_object_beverage_bottle')


def test_bottle_close_allows_only_fingers_and_next_approach_revokes_them():
    node = node_with_old_permissions()
    node._bottle_finger_contact = True
    matrix = node._target_acm('beverage_bottle')
    for link in [*node.touch_links, 'link05']:
        assert allowed(matrix, 'smolvla_object_beverage_bottle', link) == (
            link in ['gripperStator', 'gripperMover'])
    # Changing target must also revoke bottle permissions.
    matrix = node._target_acm('cup')
    assert not allowed(matrix, 'smolvla_object_beverage_bottle', 'gripperMover')
    node._bottle_finger_contact = False
    matrix = node._target_acm('beverage_bottle')
    assert not allowed(matrix, 'smolvla_object_beverage_bottle', 'gripperMover')


def test_readback_accepts_omitted_false_rows_but_rejects_wrong_permissions():
    node = node_with_old_permissions()
    expected = node._target_acm('beverage_bottle')
    import copy
    actual = copy.deepcopy(expected)
    index = actual.entry_names.index('smolvla_object_beverage_bottle')
    actual.entry_names.pop(index)
    actual.entry_values.pop(index)
    for row in actual.entry_values:
        row.enabled.pop(index)
    assert acm_permissions(actual) == acm_permissions(expected)
    node._bottle_finger_contact = True
    assert acm_permissions(actual) != acm_permissions(node._target_acm('beverage_bottle'))


@pytest.mark.parametrize('reply', [None, SimpleNamespace(success=False, message='readback mismatch')])
def test_assistant_refuses_to_continue_without_confirmed_permissions(reply):
    sent = []
    client = SimpleNamespace(wait_for_service=lambda **_: True,
                             call_async=lambda request: sent.append(request))
    assistant = SimpleNamespace(_bottle_collision_client=client,
                                 _wait_future=lambda *_: reply)
    with pytest.raises(RuntimeError, match='碰撞许可未确认'):
        GraspAssistant._set_bottle_finger_collision(assistant, True)
    assert sent[0].data is True


def test_cannot_enable_bottle_contact_when_another_object_is_selected():
    node = SimpleNamespace(_contact_service_lock=threading.Lock(),
                           _lock=threading.RLock(), _target_object='cup')
    response = SimpleNamespace(success=None, message='')
    result = ObjectCollisionSync._set_bottle_finger_collision(
        node, SimpleNamespace(data=True), response)
    assert result.success is False
