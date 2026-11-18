import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from z1_isaac_bridge import smolvla_rollout_core as rollout_core

from z1_isaac_bridge.smolvla_rollout_core import (
    JOINT_NAMES,
    JOINT_LIMITS,
    TASK_INSTRUCTIONS,
    GraspPhaseTracker,
    arm_tracking_error,
    build_report,
    camera_keys_for_features,
    classify_failure,
    execution_window,
    load_training_seeds,
    pregrasp_collision_reason,
    safe_joint_target,
    streams_are_fresh,
)
from z1_isaac_bridge.smolvla_sim_rollout import SmolVLASimRollout, parse_args


def test_binary_gripper_targets_preserve_arm_and_safe_closing_progress():
    current = np.array([0.0] * 6 + [-0.90])
    for predicted, expected in [(-0.90, -0.90), (-0.751, -0.90), (-0.75, 0.0), (-0.73, 0.0)]:
        raw = np.array([0.0] * 6 + [predicted])
        target = rollout_core.binary_gripper_target(raw)
        assert target[:6].tolist() == raw[:6].tolist()
        assert raw[-1] == predicted
        assert target[-1] == expected
    target = rollout_core.binary_gripper_target([0.0] * 6 + [-0.73])
    for _ in range(10):
        command = safe_joint_target(target, current, 0.10)
        assert np.max(np.abs(command - current)) <= 0.100000001
        current = command
    assert current[-1] == pytest.approx(0.0)
    assert rollout_core.binary_gripper_target([0.0] * 6 + [-0.73], -0.45)[-1] == -0.90
    with pytest.raises(ValueError):
        rollout_core.binary_gripper_target([0.0] * 6 + [np.nan])


def test_gripper_close_phase_ignores_small_motion_near_open_position():
    tracker = GraspPhaseTracker()
    assert not tracker.update(-0.90, -0.90, {})["gripper_commanded_closed"]
    assert not tracker.update(-0.89, -0.90, {})["gripper_commanded_closed"]
    assert tracker.update(-0.74, -0.90, {})["gripper_commanded_closed"]


def test_binary_label_encoding_and_cli():
    for predicted, expected in [(0.0, -0.90), (0.499, -0.90), (0.5, 0.0), (1.0, 0.0)]:
        raw = np.array([0.1] * 6 + [predicted])
        target = rollout_core.binary_gripper_target(raw, encoding="binary")
        assert target[-1] == expected
        assert np.array_equal(target[:6], raw[:6])
        assert raw[-1] == predicted
    args = parse_args(["--binary-gripper", "--gripper-action-encoding", "binary"])
    assert args.gripper_action_encoding == "binary"
    with pytest.raises(SystemExit):
        parse_args(["--no-binary-gripper", "--gripper-action-encoding", "binary"])


@pytest.mark.parametrize("binary", [False, True])
def test_rollout_publishes_binary_goal_with_original_prediction_in_trace(tmp_path, monkeypatch, binary):
    from z1_isaac_bridge import smolvla_sim_rollout as module
    monkeypatch.setattr(module, "VideoSet", lambda *args: SimpleNamespace(
        write=lambda images: None, close=lambda: None,
    ))
    args = parse_args([
        "--checkpoint", str(tmp_path), "--execution-rate", "60",
        "--binary-gripper" if binary else "--no-binary-gripper",
    ])
    joints = np.array([0.0] * 6 + [-0.90])
    raw = np.array([0.0] * 6 + [-0.73])
    commands = []
    goal = 0.0 if binary else -0.73

    def publish(command):
        assert np.max(np.abs(command - joints)) <= 0.100000001
        commands.append(command.copy())
        joints[:] = command
        rollout._joints = dict(zip(JOINT_NAMES, joints))

    rollout = SimpleNamespace(
        args=args, _lock=threading.RLock(), _randomization_status={},
        _joint_sim_stamp=0.0, _joints=dict(zip(JOINT_NAMES, joints)),
        _episode_seed=lambda *args: 123, _set_active=lambda value: None,
        _snapshot=lambda: ({}, joints.copy(), {"task_success": bool(joints[-1] >= goal - 1e-9)}, {}),
        _policy_batch=lambda *args: {}, _predict=lambda batch: np.array([raw]),
        _publish_grasp_phase=lambda *args: None, _publish_command=publish,
        _wait_sim_period=lambda stamp: setattr(rollout, "_joint_sim_stamp", stamp),
    )
    result = SmolVLASimRollout.run_episode(rollout, "beverage_bottle", 1, tmp_path)
    assert result["success"]
    assert commands[-1][-1] == pytest.approx(goal)
    assert all(np.array_equal(command[:6], raw[:6]) for command in commands)
    trace = [json.loads(line) for line in next(tmp_path.rglob("trace.jsonl")).read_text().splitlines()]
    assert trace
    assert all(row["policy_action"][-1] == -0.73 for row in trace)
    assert all(row["execution_target"][-1] == goal for row in trace)
    assert result["binary_gripper"] is binary


def test_unfreeze_waits_for_restored_physics():
    rollout = SimpleNamespace(_lock=threading.RLock(), _scene_grasp_reply={})
    restored = threading.Event()

    def reply(phase):
        SmolVLASimRollout._scene_grasp_callback(
            rollout, SimpleNamespace(data=json.dumps({"phase": phase}))
        )

    def publish(message):
        assert json.loads(message.data) == {
            "command": "unfreeze", "target": "beverage_bottle"
        }
        reply("UNFROZEN")

        def restore():
            restored.set()
            reply("PHYSICS_READY")

        threading.Timer(0.03, restore).start()

    rollout._scene_grasp_pub = SimpleNamespace(publish=publish)
    SmolVLASimRollout._set_target_frozen(rollout, "beverage_bottle", False)
    assert restored.is_set()


@pytest.mark.parametrize(
    ("cancelled", "action_gripper", "expected"),
    [
        (False, -0.74, ["freeze", "command", "unfreeze", "command"]),
        (False, -0.89, ["freeze", "command", "unfreeze"]),
        (True, -0.74, ["freeze", "unfreeze"]),
    ],
)
def test_episode_unfreezes_only_on_real_close_or_cancellation(
    tmp_path, monkeypatch, cancelled, action_gripper, expected
):
    from z1_isaac_bridge import smolvla_sim_rollout as module

    monkeypatch.setattr(module, "VideoSet", lambda *args: SimpleNamespace(
        write=lambda images: None, close=lambda: None,
    ))
    args = parse_args([
        "--checkpoint", str(tmp_path), "--freeze-target-until-close",
    ])
    joints = np.array([0.0] * 6 + [-0.90])
    action = np.array([0.0] * 6 + [action_gripper])
    events = []
    stop = threading.Event()
    if cancelled:
        stop.set()

    def publish(command):
        if command[-1] > -0.90:
            events.append("command")
            joints[-1] = command[-1]
            if action_gripper < -0.75:
                stop.set()

    rollout = SimpleNamespace(
        args=args, _lock=threading.RLock(), _randomization_status={},
        _joint_sim_stamp=0.0, _joints=dict(zip(JOINT_NAMES, joints)),
        _episode_seed=lambda *args: 123,
        _set_active=lambda value: None,
        _set_target_frozen=lambda target, frozen: events.append(
            "freeze" if frozen else "unfreeze"
        ),
        _snapshot=lambda: ({}, joints, {"task_success": bool(
            action_gripper >= -0.75 and joints[-1] >= -0.75
        )}, {}),
        _policy_batch=lambda *args: {},
        _predict=lambda batch: np.array([action]),
        _publish_grasp_phase=lambda *args: None,
        _publish_command=publish,
        _wait_sim_period=lambda stamp: None,
    )
    result = SmolVLASimRollout.run_episode(
        rollout, "beverage_bottle", 1, tmp_path, stop_event=stop,
    )
    assert events == expected
    assert result["freeze_target_until_close"] is True
    assert result["success"] is (not cancelled and action_gripper >= -0.75)


def test_safe_joint_target_clamps_limits_and_per_cycle_step():
    current = np.zeros(7)
    requested = np.array([9.0, 9.0, -9.0, 0.05, -0.05, 9.0, -9.0])
    result = safe_joint_target(requested, current, max_step_rad=0.10)
    assert result.tolist() == pytest.approx(
        [0.10, 0.10, -0.10, 0.05, -0.05, 0.10, -0.10]
    )


def test_safe_joint_target_never_returns_outside_joint_limits():
    current = np.array([4.0, -1.0, 1.0, 2.0, 2.0, 4.0, 1.0])
    result = safe_joint_target(np.zeros(7), current, max_step_rad=0.10)
    assert np.all(result <= JOINT_LIMITS[:, 1])
    assert np.all(result >= JOINT_LIMITS[:, 0])


@pytest.mark.parametrize(
    "requested",
    [np.zeros(6), np.array([0, 0, 0, 0, 0, 0, np.nan])],
)
def test_safe_joint_target_rejects_bad_policy_output(requested):
    with pytest.raises(ValueError):
        safe_joint_target(requested, np.zeros(7))


def test_execution_window_replans_after_first_ten_actions():
    chunk = np.arange(50 * 7).reshape(1, 50, 7)
    assert np.array_equal(execution_window(chunk, 10), chunk[0, :10])


def test_pregrasp_collision_stops_on_finger_contact_or_bottle_motion():
    baseline = (0.45, 0.0, 0.0)
    status = {"target_object": "beverage_bottle", "object_position_m": [0.45, 0.0, 0.02]}
    assert pregrasp_collision_reason(status, "beverage_bottle", -0.9, baseline) is None
    status["left_finger_contact_object"] = "beverage_bottle"
    assert pregrasp_collision_reason(status, "beverage_bottle", -0.9, baseline) == "pregrasp_target_contact"
    assert pregrasp_collision_reason(status, "beverage_bottle", -0.7, baseline) is None
    status.pop("left_finger_contact_object")
    status["object_position_m"] = [0.47, 0.0, 0.02]
    assert pregrasp_collision_reason(status, "beverage_bottle", -0.9, baseline) == "pregrasp_target_moved"


def test_stream_freshness_requires_all_inputs_within_half_second():
    stamps = {"joint": 9.8, "wrist": 9.7, "vla_wrist": 9.6, "global": 9.51}
    assert streams_are_fresh(stamps, now=10.0, timeout_s=0.5)
    stamps["wrist"] = 9.49
    assert not streams_are_fresh(stamps, now=10.0, timeout_s=0.5)


def test_two_camera_checkpoint_does_not_require_vla_wrist_stream():
    features = {
        "observation.state",
        "observation.images.wrist",
        "observation.images.global",
    }
    cameras = camera_keys_for_features(features)
    assert cameras == ("wrist", "global")
    stamps = {"joint": 9.8, "wrist": 9.7, "global": 9.6}
    assert streams_are_fresh(stamps, now=10.0, camera_keys=cameras)


def test_single_camera_act_checkpoint_uses_only_its_configured_stream():
    assert camera_keys_for_features({"observation.images.wrist"}) == ("wrist",)
    assert camera_keys_for_features({"observation.images.global"}) == ("global",)
    with pytest.raises(ValueError, match="camera"):
        camera_keys_for_features({"observation.state"})


def test_training_bounds_rejects_outside_or_missing_bottle_pose():
    assert hasattr(rollout_core, "target_in_xy_bounds")
    target_in_xy_bounds = rollout_core.target_in_xy_bounds
    bounds = (0.471597, 0.686651, -0.266489, -0.007769)
    inside = {"object_positions_m": {"beverage_bottle": [0.55, -0.12, 0.08]}}
    outside = {"object_positions_m": {"beverage_bottle": [0.55, 0.03, 0.08]}}
    assert target_in_xy_bounds(inside, "beverage_bottle", bounds)
    assert not target_in_xy_bounds(outside, "beverage_bottle", bounds)
    assert not target_in_xy_bounds({}, "beverage_bottle", bounds)


def test_covered_cell_rejects_peach_far_from_its_only_training_pose():
    rollout = SimpleNamespace(
        covered_cells={"peach": {(3, 3)}},
        training_target_positions={"peach": [(0.563763, -0.008364)]},
        args=SimpleNamespace(max_training_target_distance_m=0.03),
    )
    far = {"object_positions_m": {"peach": [0.610657, 0.059837, 0.0]}}
    near = {"object_positions_m": {"peach": [0.57, -0.01, 0.0]}}
    assert not SmolVLASimRollout._target_in_covered_cell(rollout, far, "peach")
    assert SmolVLASimRollout._target_in_covered_cell(rollout, near, "peach")


def test_snapshot_rejects_fresh_but_blank_camera():
    now = time.monotonic()
    rollout = SimpleNamespace(
        _lock=threading.Lock(),
        _stamps={"joint": now, "wrist": now, "global": now},
        _images={
            "wrist": np.full((360, 640, 3), 194, dtype=np.uint8),
            "global": np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8),
        },
        _joints=dict.fromkeys(JOINT_NAMES, 0.0),
        _status={}, _evidence={}, camera_keys=("wrist", "global"),
        args=SimpleNamespace(stale_timeout=0.5, reject_blank_cameras=True, prelift_contact_stable_s=0.0),
    )
    with pytest.raises(RuntimeError, match="blank_camera:wrist"):
        SmolVLASimRollout._snapshot(rollout)


def test_act_scene_args_keep_bottle_only_and_training_bounds():
    args = parse_args([
        "--policy-type", "act", "--checkpoint", "/home/lp531/pretrained_model",
        "--objects", "beverage_bottle", "--target-xy-bounds",
        "0.471597", "0.686651", "-0.266489", "-0.007769",
    ])
    assert args.policy_type == "act"
    assert args.objects == ["beverage_bottle"]
    assert args.target_xy_bounds == (0.471597, 0.686651, -0.266489, -0.007769)


def test_pi05_scene_args_accept_paired_eval_seeds():
    args = parse_args([
        "--policy-type", "pi05", "--checkpoint", "/tmp/checkpoint",
        "--objects", "beverage_bottle", "--episodes-per-object", "2",
        "--eval-seeds", "101", "202",
    ])
    assert args.eval_seeds == [101, 202]
    for seeds in [("101",), ("101", "101"), ("101", "-1")]:
        with pytest.raises(SystemExit):
            parse_args([
                "--policy-type", "pi05", "--checkpoint", "/tmp/checkpoint",
                "--objects", "beverage_bottle", "--episodes-per-object", "2",
                "--eval-seeds", *seeds,
            ])


def test_act_scene_args_accept_exact_paired_eval_seeds():
    args = parse_args([
        "--policy-type", "act", "--checkpoint", "/home/lp531/pretrained_model",
        "--objects", "beverage_bottle", "--target-xy-bounds",
        "0.471597", "0.686651", "-0.266489", "-0.007769",
        "--episodes-per-object", "2", "--eval-seeds", "101", "202",
    ])
    assert args.eval_seeds == [101, 202]
    with pytest.raises(SystemExit):
        parse_args([
            "--policy-type", "act", "--checkpoint", "/home/lp531/pretrained_model",
            "--objects", "beverage_bottle", "--target-xy-bounds",
            "0.471597", "0.686651", "-0.266489", "-0.007769",
            "--episodes-per-object", "2", "--eval-seeds", "101",
        ])


def test_fixed_eval_seed_resets_robot_before_replaying_scene():
    class FakePublisher:
        def __init__(self, owner):
            self.owner = owner
            self.commands = []

        def publish(self, message):
            assert self.owner.reset_done, "Objects randomized before robot reset"
            command = json.loads(message.data)
            self.commands.append(command)
            self.owner._randomization_status = {
                "phase": "RANDOMIZED",
                "seed": command["seed"],
                "object_positions_m": {"beverage_bottle": [0.6, -0.1, 0.0]},
            }

    rollout = SimpleNamespace(
        _lock=threading.Lock(), _randomization_status={}, _evidence={},
        _status={}, _seen_eval_seeds=set(), training_seeds=set(),
        args=SimpleNamespace(target_xy_bounds=(0.47, 0.69, -0.27, -0.007)),
        _set_active=lambda value: None,
        _reset_episode=object(), reset_done=False,
    )
    rollout._randomization_command_pub = FakePublisher(rollout)
    def trigger(client, timeout):
        assert client is rollout._reset_episode
        rollout.reset_done = True
    rollout._call_trigger = trigger
    rollout._target_in_covered_cell = lambda status, target: True
    assert SmolVLASimRollout._randomize_scene(
        rollout, "beverage_bottle", 123456
    ) == 123456
    assert rollout._randomization_command_pub.commands == [
        {"seed": 123456, "level": "full", "target_grid": None,
         "settle_duration_s": 2.0,
         "non_target_max_angular_speed_rad_s": 0.35}
    ]


def test_tracking_error_uses_six_arm_joints():
    current = np.zeros(7)
    command = np.array([0.01, 0.02, 0.03, 0.09, 0.04, 0.05, -1.0])
    assert arm_tracking_error(command, current) == pytest.approx(0.09)


def test_grasp_phase_tracker_waits_for_contact_before_announcing_lift():
    tracker = GraspPhaseTracker()
    closing = tracker.update(-0.60, -0.65, {"bilateral_contact": False})
    assert closing["gripper_commanded_closed"]
    assert not closing["gripper_obstructed"]
    assert not closing["physical_test_lift_active"]
    contacted = tracker.update(-0.60, -0.60, {"bilateral_contact": True})
    assert contacted["gripper_obstructed"]
    assert contacted["physical_test_lift_active"]
    tracker.update(-0.60, -0.60, {"physical_grasp_ok": True})
    opening = tracker.update(-0.70, -0.65, {"physical_grasp_ok": True})
    assert opening["release_started"]
    released = tracker.update(-0.90, -0.85, {"physical_grasp_ok": True})
    assert released["release_sequence_complete"]
    assert not released["gripper_commanded_closed"]


def test_binary_gripper_intent_survives_contact_command_fluctuation():
    tracker = GraspPhaseTracker()
    status = {"physical_grasp_ok": True, "bilateral_contact": True}
    tracker.update(-0.55, -0.65, status, gripper_target=0.0)
    holding = tracker.update(-0.57, -0.67, status, gripper_target=0.0)
    assert holding["gripper_commanded_closed"]
    assert not holding["release_started"]
    opening = tracker.update(-0.77, -0.67, status, gripper_target=-0.90)
    assert opening["release_started"]
    assert not opening["release_sequence_complete"]
    released = tracker.update(-0.90, -0.85, status, gripper_target=-0.90)
    assert released["release_sequence_complete"]


def test_task_text_and_joint_order_match_recorded_dataset():
    assert JOINT_NAMES == (
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        "jointGripper",
    )
    assert TASK_INSTRUCTIONS["beverage_bottle"] == (
        "Physically grasp the bottle, lift it clear, and place it in the box."
    )


def test_failure_classification_uses_physical_predicates():
    assert classify_failure({}, timed_out=False, stop_reason="stale_stream") == "stale_stream"
    assert classify_failure({"lifted": False}, timed_out=True) == "grasp_failed"
    assert classify_failure(
        {"physical_grasp_ok": True, "lifted": True, "release_sequence_complete": False},
        timed_out=True,
    ) == "transport_or_release_failed"
    assert classify_failure(
        {"physical_grasp_ok": True, "lifted": True, "release_sequence_complete": True},
        timed_out=True,
    ) == "placement_failed"


def test_report_has_per_object_and_overall_success_rates():
    results = [
        {"target_object": "cup", "success": True, "failure_category": "success"},
        {"target_object": "cup", "success": False, "failure_category": "grasp_failed"},
        {"target_object": "apple", "success": True, "failure_category": "success"},
    ]
    report = build_report(results)
    assert report["overall"]["success_rate"] == pytest.approx(2 / 3)
    assert report["per_object"]["cup"]["success_rate"] == 0.5
    assert report["failure_categories"] == {"grasp_failed": 1}
    assert report["acceptance"]["passed"] is False


def test_load_training_seeds_reads_selected_raw_episode_summaries(tmp_path):
    source = tmp_path / "raw"
    episode = source / "episode_000004"
    episode.mkdir(parents=True)
    (episode / "episode_summary.json").write_text(
        json.dumps({"scene_randomization": {"seed": 1234}}), encoding="utf-8"
    )
    manifest = tmp_path / "selection_manifest.json"
    manifest.write_text(json.dumps({
        "source": str(source), "objects": {"cup": [episode.name]}
    }), encoding="utf-8")
    assert load_training_seeds(manifest) == {1234}


def test_rollout_waits_for_isaac_services_when_started_from_launch(tmp_path):
    args = parse_args(["--checkpoint", str(tmp_path / "checkpoint")])
    assert args.startup_timeout == 300.0


def test_prepared_episode_seed_skips_scene_preparation():
    class FakeRollout:
        calls = 0

        def _prepare_episode(self, target):
            self.calls += 1
            return 777

    rollout = FakeRollout()
    assert SmolVLASimRollout._episode_seed(rollout, "cup", 123) == 123
    assert rollout.calls == 0
    assert SmolVLASimRollout._episode_seed(rollout, "cup", None) == 777
    assert rollout.calls == 1


def test_fixed_eval_seed_uses_scene_preparation():
    class FakeRollout:
        def _prepare_episode(self, target, fixed_seed=None):
            assert target == "beverage_bottle"
            assert fixed_seed == 123
            return fixed_seed

    assert SmolVLASimRollout._episode_seed(
        FakeRollout(), "beverage_bottle", None, 123
    ) == 123
    with pytest.raises(ValueError, match="seed"):
        SmolVLASimRollout._episode_seed(
            FakeRollout(), "beverage_bottle", 123, 123
        )


def test_rollout_supports_operator_stop_during_execution():
    import inspect

    signature = inspect.signature(SmolVLASimRollout.run_episode)
    assert "prepared_seed" in signature.parameters
    assert "stop_event" in signature.parameters
    source = inspect.getsource(SmolVLASimRollout.run_episode)
    assert source.count('stop_reason = "operator_stop"') >= 2
    assert classify_failure({}, timed_out=False, stop_reason="operator_stop") == (
        "operator_stop"
    )


def test_three_camera_pi05_requires_right_external_stream():
    cameras = camera_keys_for_features({
        'observation.state', 'observation.images.wrist',
        'observation.images.global', 'observation.images.global_right',
    })
    assert cameras == ('wrist', 'global', 'global_right')
    stamps = {'joint': 10., 'wrist': 10., 'global': 10.}
    assert not streams_are_fresh(stamps, now=10., camera_keys=cameras)
    stamps['global_right'] = 10.
    assert streams_are_fresh(stamps, now=10., camera_keys=cameras)


def test_binary_gripper_hysteresis_holds_across_midpoint_and_releases():
    previous = -0.9
    outputs = []
    for value in [0.48, 0.52, 0.6, 0.49, 0.54, 0.41, 0.4, 0.51]:
        result = rollout_core.binary_gripper_target(
            [0.1] * 6 + [value], encoding='binary', previous_target=previous)
        np.testing.assert_array_equal(result[:6], [0.1] * 6)
        previous = result[-1]
        outputs.append(previous)
    assert outputs == [-0.9, -0.9, 0., 0., 0., 0., -0.9, -0.9]


def test_angle_encoding_retains_original_threshold_with_previous_target():
    result = rollout_core.binary_gripper_target(
        [0.] * 6 + [-0.73], encoding='angle', previous_target=-0.9)
    assert result[-1] == 0.
