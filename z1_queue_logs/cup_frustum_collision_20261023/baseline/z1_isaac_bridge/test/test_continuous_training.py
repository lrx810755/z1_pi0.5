from z1_isaac_bridge.continuous_training import ContinuousTrainingRunner


class FakeApi:
    def __init__(self, demonstrations):
        self.demonstrations = iter(demonstrations)
        self.calls = []

    def get(self, path, timeout_s=0.0):
        self.calls.append(("GET", path))
        return {
            "connected": True,
            "recording": {"ready": True},
            "episode_reset": {"ready": True},
            "collision_scene": {"ready": True},
            "object_collisions": {"ready": True},
        }

    def post(self, path, payload=None, timeout_s=0.0):
        self.calls.append(("POST", path, payload))
        if path == "/api/demonstration/auto":
            result = next(self.demonstrations)
            if isinstance(result, Exception):
                raise result
            return result
        return {"success": True, "message": path}


def make_runner(api, episodes, target_grid=None):
    return ContinuousTrainingRunner(
        api,
        target="apple",
        curriculum="full" if target_grid is not None else "easy",
        episodes=episodes,
        target_grid=target_grid,
        retry_delay_s=0.0,
        output=lambda _message: None,
        sleep=lambda _seconds: None,
    )


def test_counts_only_successfully_saved_demonstrations():
    api = FakeApi(
        [
            RuntimeError("grasp failed"),
            {"success": True, "message": "saved 1"},
            RuntimeError("place failed"),
            {"success": True, "message": "saved 2"},
        ]
    )
    runner = make_runner(api, episodes=2)

    assert runner.run() == 0
    assert runner.successes == 2
    assert runner.failures == 2
    assert runner.attempts == 4


def test_every_failure_stops_discards_and_resets_before_retry():
    api = FakeApi(
        [RuntimeError("failed"), {"success": True, "message": "saved"}]
    )
    runner = make_runner(api, episodes=1)

    assert runner.run() == 0
    paths = [call[1] for call in api.calls]
    assert paths == [
        "/api/status",
        "/api/demonstration/auto",
        "/api/force_stop",
        "/api/recording/discard",
        "/api/reset_episode",
        "/api/demonstration/auto",
    ]


def test_max_attempts_stops_without_counting_failed_episode():
    api = FakeApi([RuntimeError("failed")])
    runner = make_runner(api, episodes=1)
    runner.max_attempts = 1

    try:
        runner.run()
    except RuntimeError as exc:
        assert "最大尝试次数" in str(exc)
    else:
        raise AssertionError("runner should enforce max_attempts")
    assert runner.successes == 0
    assert runner.failures == 1


def test_grid_lock_is_sent_on_every_retry():
    api = FakeApi(
        [RuntimeError("failed"), {"success": True, "message": "saved"}]
    )
    runner = make_runner(api, episodes=1, target_grid=(2, 3))

    assert runner.run() == 0
    requests = [
        call[2]
        for call in api.calls
        if call[0:2] == ("POST", "/api/demonstration/auto")
    ]
    assert requests == [
        {
            "target_object": "apple",
            "curriculum_level": "full",
            "target_grid": [2, 3],
        },
        {
            "target_object": "apple",
            "curriculum_level": "full",
            "target_grid": [2, 3],
        },
    ]


def test_grid_lock_requires_full_curriculum():
    api = FakeApi([])
    try:
        ContinuousTrainingRunner(
            api,
            target="apple",
            curriculum="easy",
            episodes=1,
            target_grid=(0, 0),
        )
    except ValueError as exc:
        assert "full curriculum" in str(exc)
    else:
        raise AssertionError("grid lock must not be silently ignored")
