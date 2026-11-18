"""Bounded simulation-time buffers for fixed-rate demonstration recording."""

from bisect import bisect_right
from collections import deque
from math import isfinite


class RecordingSynchronizer:
    def __init__(self, cameras, tolerance=0.025, state_gap=0.1,
                 action_age=0.1, capacity=256):
        self.cameras = tuple(cameras)
        self.tolerance = tolerance
        self.state_gap = state_gap
        self.action_age = action_age
        self.buffers = {
            name: deque(maxlen=min(capacity, 64) if name in self.cameras else capacity)
            for name in (*self.cameras, "state", "action", "status")
        }
        self.used_images = {}

    def push(self, name, stamp, payload):
        if not isfinite(stamp) or stamp < 0:
            return
        buffer = self.buffers[name]
        # DDS may deliver out of order; never let an older message replace
        # the newest state. Clock resets are handled explicitly by the node.
        if buffer and stamp <= buffer[-1][0]:
            return
        buffer.append((stamp, payload))

    def reset(self):
        for buffer in self.buffers.values():
            buffer.clear()
        self.used_images.clear()

    def begin_episode(self):
        self.used_images.clear()

    def select_observation(self, target):
        """Pair images and interpolate measured joints at one sensor time."""
        images = {}
        for name in self.cameras:
            candidates = [
                sample for sample in self.buffers[name]
                if abs(sample[0] - target) <= self.tolerance + 1e-9
                and sample[0] > self.used_images.get(name, -1)
            ]
            if not candidates:
                raise ValueError(f"{name}: no fresh image near {target:.6f}")
            images[name] = min(candidates, key=lambda sample: abs(sample[0] - target))

        stamps = [target, *(sample[0] for sample in images.values())]
        if max(stamps) - min(stamps) > self.tolerance + 1e-9:
            raise ValueError("camera/state timestamp spread exceeds limit")

        states = list(self.buffers["state"])
        stamps = [sample[0] for sample in states]
        index = bisect_right(stamps, target)
        if index and abs(states[index - 1][0] - target) < 1e-9:
            left = right = states[index - 1]
            state = list(left[1])
        elif index == 0 or index == len(states):
            raise ValueError("state: interpolation requires bracketing samples")
        else:
            left, right = states[index - 1], states[index]
            if right[0] - left[0] > self.state_gap + 1e-9:
                raise ValueError("state: interpolation gap exceeds limit")
            fraction = (target - left[0]) / (right[0] - left[0])
            state = [a + fraction * (b - a) for a, b in zip(left[1], right[1])]

        return {
            "time": target, "images": images, "state": state,
            "state_bracket": (left[0], right[0]),
        }

    def select_latest_observation(self, now, max_age):
        """Choose the newest unused, bounded-age camera/state observation."""
        targets = sorted({
            stamp for name in self.cameras for stamp, _ in self.buffers[name]
            if now - max_age - 1e-9 <= stamp <= now + 1e-9
        }, reverse=True)
        for target in targets:
            try:
                sample = self.select_observation(target)
            except ValueError:
                continue
            stamps = [target, *(value[0] for value in sample["images"].values())]
            if (max(stamps) - min(stamps) <= self.tolerance + 1e-9
                    and now - min(stamps) <= max_age + 1e-9
                    and max(stamps) <= now + 1e-9):
                return sample
        raise ValueError("no fresh camera/state observation within sync tolerance and age limit")

    def select(self, target):
        sample = self.select_observation(target)
        commands = [sample for sample in self.buffers["action"] if sample[0] <= target]
        if not commands or target - commands[-1][0] > self.action_age + 1e-9:
            stamps = [sample[0] for sample in self.buffers["action"]]
            raise ValueError(
                "action: no recent causal command "
                f"(target={target:.6f}, count={len(stamps)}, "
                f"oldest={stamps[0] if stamps else None}, "
                f"newest={stamps[-1] if stamps else None})"
            )
        statuses = [sample for sample in self.buffers["status"] if sample[0] <= target]
        if not statuses or target - statuses[-1][0] > 1.0:
            raise ValueError("status: no recent task status")
        return {
            **sample,
            "action": commands[-1], "status": statuses[-1][1],
        }

    def commit(self, sample):
        self.used_images.update({name: value[0] for name, value in sample["images"].items()})
