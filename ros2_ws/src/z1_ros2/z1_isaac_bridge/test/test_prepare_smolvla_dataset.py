from z1_isaac_bridge.prepare_smolvla_dataset import clean_frames


def _frame(index, state_value, action_value, phase="READY"):
    return {
        "frame_index": index,
        "timestamp": index / 10,
        "observation.state": [state_value] * 7,
        "action": [action_value] * 7,
        "phase": phase,
        "observation.images.wrist": f"wrist/{index}.jpg",
        "observation.images.vla_wrist": f"vla_wrist/{index}.jpg",
        "observation.images.global": f"global/{index}.jpg",
    }


def test_clean_frames_keeps_three_frame_preroll_before_arm_motion():
    frames = [
        _frame(0, 0.0, 0.0),
        _frame(1, 0.0, 0.5),  # An action/gripper change is not arm motion.
        _frame(2, 0.0, 0.5),
        _frame(3, 0.0, 0.5),
        _frame(4, 0.0, 0.5),
        _frame(5, 0.003, 0.5),
        _frame(6, 0.004, 0.5),
    ]

    cleaned, removed = clean_frames(
        frames, static_threshold=0.002, leading_context_frames=3
    )

    assert removed == 2
    assert [frame["frame_index"] for frame in cleaned] == list(range(5))
    assert [frame["timestamp"] for frame in cleaned] == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert [frame["observation.state"][0] for frame in cleaned] == [0.0, 0.0, 0.0, 0.003, 0.004]
    assert all("observation.images.vla_wrist" not in frame for frame in cleaned)


def test_clean_frames_limits_actions_and_makes_phases_monotonic():
    frames = [
        _frame(0, 0.0, -0.9, "READY"),
        _frame(1, 0.003, 0.0, "GRASPING"),
        _frame(2, 0.004, 0.0, "PHYSICAL_TEST_LIFT"),
        _frame(3, 0.005, 0.0, "READY"),
        _frame(4, 0.006, -0.9, "PLACING"),
        _frame(5, 0.006, -0.9, "SUCCESS"),
        _frame(6, 0.006, -0.9, "SUCCESS"),
    ]

    cleaned, _ = clean_frames(
        frames,
        static_threshold=0.002,
        leading_context_frames=1,
        max_action_step_rad=0.1,
        terminal_hold_frames=1,
    )

    assert [frame["phase"] for frame in cleaned] == [
        "READY",
        "GRASPING",
        "PHYSICAL_TEST_LIFT",
        "PHYSICAL_TEST_LIFT",
        "PLACING",
        "SUCCESS",
    ]
    assert max(
        abs(frame["action"][joint] - frame["observation.state"][joint])
        for frame in cleaned
        for joint in range(7)
    ) <= 0.100001


def test_successful_episode_keeps_recorded_transport_despite_gripper_dip():
    frames = [
        _frame(0, 0.0, -0.9, "READY"),
        _frame(1, 0.003, -0.65, "GRASPING"),
        _frame(2, 0.004, -0.72, "GRASPING"),
        _frame(3, 0.005, -0.67, "PHYSICAL_TEST_LIFT"),
        _frame(4, 0.006, -0.67, "TRANSPORT"),
        _frame(5, 0.007, -0.9, "PLACING"),
        _frame(6, 0.008, -0.9, "SUCCESS"),
        _frame(7, 0.009, -0.9, "SUCCESS"),
        _frame(8, 0.010, -0.9, "SUCCESS"),
    ]

    cleaned, _ = clean_frames(
        frames,
        leading_context_frames=1,
        terminal_hold_frames=2,
    )

    assert [frame["phase"] for frame in cleaned] == [
        "READY", "GRASPING", "GRASPING", "PHYSICAL_TEST_LIFT",
        "TRANSPORT", "PLACING", "SUCCESS", "SUCCESS",
    ]


def test_long_identical_holds_are_capped_but_success_tail_is_kept():
    frames = [_frame(0, 0.0, -0.9)]
    frames += [_frame(i, 0.003, 0.0, "GRASPING") for i in range(1, 9)]
    frames += [_frame(i, 0.003, -0.9, "SUCCESS") for i in range(9, 14)]

    cleaned, _ = clean_frames(
        frames,
        leading_context_frames=1,
        terminal_hold_frames=5,
        max_identical_run_frames=3,
    )

    assert [frame["phase"] for frame in cleaned].count("GRASPING") == 3
    assert [frame["phase"] for frame in cleaned].count("SUCCESS") == 5
    assert [frame["frame_index"] for frame in cleaned] == list(range(len(cleaned)))
