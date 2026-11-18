import pytest
from z1_isaac_bridge import smolvla_rollout_core as core


def test_close_hold_blocks_early_open_without_extending_deadline():
    hold=core.GripperCloseHold(1.)
    assert hold.apply(-.9,10.) == -.9
    assert hold.apply(0.,11.) == 0.
    assert hold.apply(0.,11.5) == 0.
    assert hold.apply(-.9,11.9) == 0.
    assert hold.apply(-.9,12.) == -.9
    assert hold.apply(0.,13.) == 0.
    assert hold.apply(-.9,13.9) == 0.
    assert hold.apply(-.9,14.) == -.9


def test_new_episode_and_disabled_hold_allow_open():
    first=core.GripperCloseHold(1.);first.apply(0.,20.)
    assert core.GripperCloseHold(1.).apply(-.9,20.1)==-.9
    disabled=core.GripperCloseHold(0.);disabled.apply(0.,20.)
    assert disabled.apply(-.9,20.)==-.9


@pytest.mark.parametrize('duration',[-1.,float('nan'),float('inf')])
def test_invalid_hold_duration(duration):
    with pytest.raises(ValueError):core.GripperCloseHold(duration)
