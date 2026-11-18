"""Run the user-local MoveIt Servo binary without requiring sudo."""

from __future__ import annotations

import os
from pathlib import Path
import sys


SERVO_PREFIX = Path(
    "/home/lp531/.local/opt/ros-humble-moveit-servo/opt/ros/humble"
)
SERVO_BINARY = SERVO_PREFIX / "lib/moveit_servo/servo_node_main"


def main() -> None:
    if not SERVO_BINARY.is_file():
        raise SystemExit(
            "MoveIt Servo user-local binary is missing: " + str(SERVO_BINARY)
        )
    environment = dict(os.environ)
    library_path = str(SERVO_PREFIX / "lib")
    current = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        library_path if not current else library_path + ":" + current
    )
    os.execve(
        str(SERVO_BINARY),
        [str(SERVO_BINARY), *sys.argv[1:]],
        environment,
    )


if __name__ == "__main__":
    main()
