"""Pure command and state-machine helpers for SmolVLA voice control."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Mapping


TARGET_ALIASES = {
    "cup": ("cup", "杯子", "杯"),
    "apple": ("apple", "苹果"),
    "peach": ("peach", "桃子", "桃"),
    "beverage_bottle": (
        "beverage_bottle", "beverage bottle", "bottle", "饮料瓶", "水瓶", "瓶子", "瓶",
    ),
}
VALID_TARGETS = frozenset(TARGET_ALIASES)


@dataclass(frozen=True)
class VoiceCommand:
    command: str
    target: str | None = None


class InvalidTransition(RuntimeError):
    pass


def match_target(text: str, parsed_object: str | None = None) -> str | None:
    haystack = f"{text or ''} {parsed_object or ''}".lower()

    def contains(alias: str) -> bool:
        if alias.isascii():
            return re.search(
                rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", haystack
            ) is not None
        return alias in haystack

    matches = {
        target
        for target, aliases in TARGET_ALIASES.items()
        if any(contains(alias) for alias in aliases)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def decode_command(payload: str | Mapping[str, object]) -> VoiceCommand:
    try:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("voice command must be a JSON object") from exc
    if not isinstance(data, Mapping):
        raise ValueError("voice command must be a JSON object")
    command = data.get("command")
    target = data.get("target")
    if command in {"start", "stop"} and target is None:
        return VoiceCommand(str(command))
    if command == "grasp" and target in VALID_TARGETS:
        return VoiceCommand("grasp", str(target))
    raise ValueError("unsupported voice command")


class VoiceSession:
    def __init__(self) -> None:
        self.state = "IDLE"
        self.seed: int | None = None
        self.target: str | None = None
        self.stop_requested = False

    def accept(self, command: VoiceCommand) -> str:
        if command.command == "start" and self.state == "IDLE":
            self.state = "RANDOMIZING"
            self.stop_requested = False
            return "randomize"
        if command.command == "grasp" and self.state == "LISTENING":
            self.state = "EXECUTING"
            self.target = command.target
            self.stop_requested = False
            return "execute"
        if command.command == "stop" and self.state == "LISTENING":
            self._reset()
            return "cancel"
        if command.command == "stop" and self.state == "EXECUTING":
            self.stop_requested = True
            return "stop"
        raise InvalidTransition(f"cannot {command.command} while {self.state}")

    def randomized(self, seed: int) -> None:
        if self.state != "RANDOMIZING":
            raise InvalidTransition(f"cannot finish randomization while {self.state}")
        self.seed = int(seed)
        self.state = "LISTENING"

    def randomization_failed(self) -> None:
        if self.state != "RANDOMIZING":
            raise InvalidTransition(f"cannot fail randomization while {self.state}")
        self._reset()

    def episode_done(self) -> None:
        if self.state != "EXECUTING":
            raise InvalidTransition(f"cannot finish episode while {self.state}")
        self._reset()

    def _reset(self) -> None:
        self.state = "IDLE"
        self.seed = None
        self.target = None
        self.stop_requested = False
