"""Shared state schema for Z1 SmolVLA datasets and live rollout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


JOINT_STATE_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "jointGripper",
)
PHASES = (
    "READY",
    "GRASPING",
    "PHYSICAL_TEST_LIFT",
    "TRANSPORT",
    "PLACING",
    "SUCCESS",
)
AUGMENTED_STATE_NAMES = (
    *JOINT_STATE_NAMES,
    *(f"phase.{phase.lower()}" for phase in PHASES),
)


def augmented_state(
    joints: Sequence[float], status: Mapping[str, object]
) -> tuple[float, ...]:
    """Encode joints and the one-hot task phase."""
    if len(joints) != len(JOINT_STATE_NAMES):
        raise ValueError(f"expected 7 joints, got {len(joints)}")
    joint_values = tuple(float(value) for value in joints)
    if not all(math.isfinite(value) for value in joint_values):
        raise ValueError("joint state contains a non-finite value")
    phase = str(status.get("phase", "")).strip().upper()
    if phase not in PHASES:
        raise ValueError(f"unknown SmolVLA phase: {phase!r}")
    phase_values = tuple(float(name == phase) for name in PHASES)
    return joint_values + phase_values
