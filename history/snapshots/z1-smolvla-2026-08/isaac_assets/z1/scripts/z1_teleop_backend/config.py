import os
from pathlib import Path

import numpy as np


def _parse_vec3_literal(name, default):
    raw = os.environ.get(name)
    if not raw:
        return np.asarray(default, dtype=np.float64)
    try:
        values = [float(item.strip()) for item in raw.split(",")]
        if len(values) != 3:
            raise ValueError
        return np.asarray(values, dtype=np.float64)
    except Exception:
        print(f"[WARN] {name}={raw!r} 解析失败，继续使用默认值 {default}。")
        return np.asarray(default, dtype=np.float64)


REPO_ROOT = Path(
    os.environ.get(
        "Z1_SMOLVLA_ROOT",
        Path(__file__).resolve().parents[4],
    )
).expanduser().resolve()
ROOT_DIR = Path(
    os.environ.get("Z1_ASSET_ROOT", REPO_ROOT / "isaac_assets" / "z1")
).expanduser().resolve()
ISAAC_SIM_ROOT = Path(
    os.environ.get("ISAAC_SIM_ROOT", Path.home() / "isaac-sim")
).expanduser().resolve()
USD_PATH = ROOT_DIR / "z1.usd"
URDF_PATH = ROOT_DIR / "z1.urdf"
HAND_EYE_CALIBRATION_PATH = ROOT_DIR / "hand_eye_calibration.yaml"
FRONTEND_PATH = ROOT_DIR / "scripts" / "z1_teleop_frontend.html"
ROBOT_STAGE_PATH = "/World/z1_debug"
HOST = "127.0.0.1"
PORT = int(os.environ.get("Z1_TELEOP_PORT", "8765"))
SMOLVLA_GUI_BRIDGE_URL = os.environ.get("Z1_SMOLVLA_GUI_BRIDGE_URL", "http://127.0.0.1:8767")
MAX_JSON_BODY_BYTES = int(os.environ.get("Z1_MAX_JSON_BODY_BYTES", "65536"))

REAL_ARM_ENABLED = os.environ.get("Z1_REAL_ARM_ENABLED", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
REAL_ARM_HOST = os.environ.get("Z1_REAL_ARM_HOST", "127.0.0.1")
REAL_ARM_PORT = int(os.environ.get("Z1_REAL_ARM_PORT", "12001"))
REAL_ARM_URL = os.environ.get("Z1_REAL_ARM_URL", f"http://{REAL_ARM_HOST}:{REAL_ARM_PORT}")
REAL_ARM_SYNC_INTERVAL_S = float(os.environ.get("Z1_REAL_ARM_SYNC_INTERVAL_S", "0.03"))
REAL_ARM_REQUEST_TIMEOUT_S = float(os.environ.get("Z1_REAL_ARM_REQUEST_TIMEOUT_S", "0.35"))
REAL_ARM_STARTUP_TIMEOUT_S = float(os.environ.get("Z1_REAL_ARM_STARTUP_TIMEOUT_S", "8.0"))
REAL_ARM_MANAGE_CONTROLLER = os.environ.get("Z1_REAL_ARM_MANAGE_CONTROLLER", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
REAL_ARM_MANAGE_BRIDGE = os.environ.get("Z1_REAL_ARM_MANAGE_BRIDGE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
REAL_ARM_ROBOT_IP = os.environ.get("Z1_REAL_ARM_IP", "192.168.123.110")
REAL_ARM_CONTROLLER_DIR = Path(
    os.environ.get(
        "Z1_REAL_ARM_CONTROLLER_DIR",
        Path.home() / "unitree_z1" / "z1_controller",
    )
)
REAL_ARM_CONTROLLER_BIN = Path(
    os.environ.get("Z1_REAL_ARM_CONTROLLER_BIN", str(REAL_ARM_CONTROLLER_DIR / "build" / "z1_ctrl"))
)
REAL_ARM_SDK_DIR = Path(
    os.environ.get(
        "Z1_REAL_ARM_SDK_DIR",
        Path.home() / "unitree_z1" / "z1_sdk",
    )
)
REAL_ARM_BRIDGE_PATH = Path(
    os.environ.get("Z1_REAL_ARM_BRIDGE_PATH", str(Path(__file__).resolve().parents[1] / "z1_real_arm_bridge.py"))
)
REAL_ARM_PYTHON = os.environ.get("Z1_REAL_ARM_PYTHON", "python3")
REAL_ARM_WATCHDOG_S = float(os.environ.get("Z1_REAL_ARM_WATCHDOG_S", "0.45"))
REAL_ARM_JOINT_SPEED_RAD_S = float(os.environ.get("Z1_REAL_ARM_JOINT_SPEED_RAD_S", "0.6"))

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "jointGripper"]
MOVEIT_ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
MOVEIT_LOCKED_JOINTS = [
    name.strip()
    for name in os.environ.get("Z1_MOVEIT_LOCKED_JOINTS", "").split(",")
    if name.strip()
]
MOVEIT_WARN_JOINT_DELTAS = {"joint5": float(os.environ.get("Z1_MOVEIT_JOINT5_WARN_DELTA_RAD", "0.05"))}
WRIST_CAMERA_NAME = "WristCamera"
CAMERA_VISUAL_ENABLED = os.environ.get("Z1_CAMERA_VISUAL_ENABLED", "0").strip().lower() not in ("0", "false", "no", "off")
CAMERA_RED_VISUAL_CLEANUP_ENABLED = os.environ.get("Z1_CAMERA_RED_VISUAL_CLEANUP_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
RUNTIME_WRIST_CAMERA_ENABLED = os.environ.get("Z1_RUNTIME_WRIST_CAMERA_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
RUNTIME_WRIST_CAMERA_FORWARD_OFFSET_M = float(os.environ.get("Z1_RUNTIME_WRIST_CAMERA_FORWARD_OFFSET_M", "0.0"))
WRIST_CAMERA_LOCAL_TRANSLATION = np.array([0.055, 0.0, -0.055], dtype=np.float64)
WRIST_CAMERA_LOCAL_ROTATE_XYZ_DEG = np.array([90.0, 0.0, -90.0], dtype=np.float64)
WRIST_CAMERA_LOOK_TARGET = np.array([0.35, 0.0, 0.08], dtype=np.float64)
WRIST_CAMERA_RESOLUTION = (640, 360)
DISPLAY_CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAME_INTERVAL_S = 0.1
RED_BALL_PATH = "/World/CameraTarget"
RED_BALL_ENABLED = os.environ.get("Z1_RED_BALL_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
RED_OBJECT_GEOMETRY = os.environ.get("Z1_RED_OBJECT_GEOMETRY", "cuboid").strip().lower()
RED_OBJECT_SIZE_M = _parse_vec3_literal("Z1_RED_OBJECT_SIZE", [0.035, 0.035, 0.035])
RED_BALL_RADIUS_M = 0.04
RED_BALL_MASS_KG = float(os.environ.get("Z1_RED_BALL_MASS_KG", "0.02"))
RED_BALL_DEFAULT_POSITION = WRIST_CAMERA_LOOK_TARGET.copy()
RED_BALL_DISABLE_GRAVITY = os.environ.get("Z1_RED_BALL_DISABLE_GRAVITY", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
RED_BALL_MIN_AREA_PX = int(os.environ.get("Z1_RED_BALL_MIN_AREA_PX", "80"))
RED_BALL_MIN_RADIUS_PX = float(os.environ.get("Z1_RED_BALL_MIN_RADIUS_PX", "8.0"))
RED_BALL_EDGE_MARGIN_PX = int(os.environ.get("Z1_RED_BALL_EDGE_MARGIN_PX", "8"))
RED_BALL_MIN_CIRCULARITY = float(os.environ.get("Z1_RED_BALL_MIN_CIRCULARITY", "0.55"))
RED_BALL_MIN_COMPLETE_FRAMES = int(os.environ.get("Z1_RED_BALL_MIN_COMPLETE_FRAMES", "3"))
GRIPPER_CLOSED = -1.25
GRIPPER_OPEN = -0.15
GRASP_PREGRASP_DISTANCE_M = float(os.environ.get("Z1_GRASP_PREGRASP_DISTANCE_M", "0.22"))
# MoveIt plans link06. The Z1 gripper contact region is roughly 0.10m in
# front of link06: link06->gripperStator 0.051m, gripperStator->gripperMover
# 0.049m in the official URDF.
GRASP_APPROACH_OFFSET_M = float(os.environ.get("Z1_GRASP_APPROACH_OFFSET_M", "0.10"))
GRASP_LIFT_DISTANCE_M = float(os.environ.get("Z1_GRASP_LIFT_DISTANCE_M", "0.16"))
GRASP_MIN_GRASP_Z_M = float(os.environ.get("Z1_GRASP_MIN_GRASP_Z_M", str(RED_BALL_RADIUS_M + 0.015)))
GRASP_MIN_PREGRASP_Z_M = float(os.environ.get("Z1_GRASP_MIN_PREGRASP_Z_M", "0.18"))
GRASP_USE_PHYSICS_TARGET_FALLBACK = os.environ.get("Z1_GRASP_USE_PHYSICS_TARGET_FALLBACK", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
GRASP_PHYSICS_FALLBACK_SETTLE_STEPS = int(os.environ.get("Z1_GRASP_PHYSICS_FALLBACK_SETTLE_STEPS", "10"))
GRASP_USE_PHYSICS_FIXED_JOINT = os.environ.get("Z1_GRASP_USE_PHYSICS_FIXED_JOINT", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
GRASP_ATTACH_MAX_DISTANCE_M = float(os.environ.get("Z1_GRASP_ATTACH_MAX_DISTANCE_M", "0.12"))
GRASP_LIFT_VERIFY_MIN_DELTA_Z_M = float(os.environ.get("Z1_GRASP_LIFT_VERIFY_MIN_DELTA_Z_M", "0.05"))
GRASP_CLOSE_SETTLE_STEPS = int(os.environ.get("Z1_GRASP_CLOSE_SETTLE_STEPS", "90"))
GRASP_ASSISTED_ATTACHMENT_ENABLED = os.environ.get("Z1_GRASP_ASSISTED_ATTACHMENT_ENABLED", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
MOVEIT_BRIDGE_URL = os.environ.get("Z1_MOVEIT_BRIDGE_URL", "http://127.0.0.1:8766/plan_execute")
MOVEIT_BRIDGE_TIMEOUT_S = float(os.environ.get("Z1_MOVEIT_BRIDGE_TIMEOUT_S", "20.0"))
SAFE_HOME = {
    "joint1": 0.0,
    "joint2": 1.0,
    "joint3": -1.2,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "jointGripper": -0.4,
}
DEFAULT_SPEED_SCALE = 0.5
MAX_SAFE_SPEED_RAD_S = 2.0
MIN_LIMIT_MARGIN = 1e-4
FIXED_DEMO_BALL_POSITION = np.array([0.279, -0.147, 0.268], dtype=np.float64)
FIXED_DEMO_SUPPORT_PATH = "/World/Z1FixedDemoSupport"
FIXED_DEMO_SUPPORT_ENABLED = os.environ.get("Z1_FIXED_DEMO_SUPPORT_ENABLED", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
FIXED_DEMO_SUPPORT_SIZE_M = _parse_vec3_literal("Z1_FIXED_DEMO_SUPPORT_SIZE", [0.12, 0.12, 0.19])
FIXED_DEMO_APPROACH_TARGETS = {
    "joint1": -0.2157,
    "joint2": 1.8838,
    "joint3": -0.732,
    "joint4": -0.1405,
    "joint5": -0.7398,
    "joint6": -0.4093,
    "jointGripper": GRIPPER_OPEN,
}
FIXED_DEMO_LIFT_TARGETS = {
    "joint1": -0.2157,
    "joint2": 1.40,
    "joint3": -1.00,
    "joint4": -0.1405,
    "joint5": -0.7398,
    "joint6": -0.4093,
    "jointGripper": GRIPPER_CLOSED,
}
FIXED_DEMO_APPROACH_STEPS = int(os.environ.get("Z1_FIXED_DEMO_APPROACH_STEPS", "520"))
FIXED_DEMO_CLOSE_STEPS = int(os.environ.get("Z1_FIXED_DEMO_CLOSE_STEPS", "260"))
FIXED_DEMO_LIFT_STEPS = int(os.environ.get("Z1_FIXED_DEMO_LIFT_STEPS", "260"))
FIXED_DEMO_ATTACH_MAX_DISTANCE_M = float(os.environ.get("Z1_FIXED_DEMO_ATTACH_MAX_DISTANCE_M", "0.18"))
FIXED_DEMO_USE_FIXED_JOINT = os.environ.get("Z1_FIXED_DEMO_USE_FIXED_JOINT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
FIXED_DEMO_MIN_LIFT_DELTA_M = float(os.environ.get("Z1_FIXED_DEMO_MIN_LIFT_DELTA_M", "0.04"))
FIXED_DEMO_ALLOW_ASSIST = os.environ.get("Z1_FIXED_DEMO_ALLOW_ASSIST", "0").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
TELEOP_KP = {
    "joint1": 5000.0,
    "joint2": 10000.0,
    "joint3": 9000.0,
    "joint4": 5000.0,
    "joint5": 2500.0,
    "joint6": 1200.0,
    "jointGripper": 300.0,
}
TELEOP_KD = {
    "joint1": 400.0,
    "joint2": 800.0,
    "joint3": 700.0,
    "joint4": 400.0,
    "joint5": 200.0,
    "joint6": 100.0,
    "jointGripper": 20.0,
}
TELEOP_MAX_EFFORT = {
    "joint1": 1000.0,
    "joint2": 2000.0,
    "joint3": 1600.0,
    "joint4": 1000.0,
    "joint5": 600.0,
    "joint6": 300.0,
    "jointGripper": 80.0,
}


# =========================
# Camera options
# =========================
# The backend is intentionally based on the original working teleop script.
# It does not change the robot's initial pose, and it does not add a collision
# guard that may prevent enabling.
#
# Eye-in-hand policy for grasping:
#   - Mount the camera on gripperStator by default, not gripperMover.
#     gripperMover is a moving finger; a real wrist camera should be attached
#     to the fixed wrist/palm base below the gripper.
#   - The USD/Isaac camera optical axis is local -Z.
#   - Z1 gripper geometry extends along mount local +X, so the default
#     RotateXYZ=(90,0,-90) points camera local -Z along mount local +X.
#   - Adjust Z1_WRIST_CAMERA_TRANSLATION or Z1_WRIST_CAMERA_ROTATE_XYZ_DEG
#     if the imported mesh frame differs on your machine.


def parse_vec3_env(name, default):
    raw = os.environ.get(name)
    if not raw:
        return np.asarray(default, dtype=np.float64)
    try:
        values = [float(item.strip()) for item in raw.split(",")]
        if len(values) != 3:
            raise ValueError
        return np.asarray(values, dtype=np.float64)
    except Exception:
        print(f"[WARN] {name}={raw!r} 解析失败，继续使用默认值 {default}。")
        return np.asarray(default, dtype=np.float64)


def parse_axis_env(name, default):
    axis = os.environ.get(name, default).strip().upper().replace(" ", "")
    table = {
        "+X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }
    if axis not in table:
        print(f"[WARN] {name}={axis!r} 无效，继续使用默认值 {default}。可选: +X -X +Y -Y +Z -Z")
        axis = default
    return table[axis], axis


WRIST_CAMERA_LOCAL_TRANSLATION = parse_vec3_env(
    "Z1_WRIST_CAMERA_TRANSLATION",
    WRIST_CAMERA_LOCAL_TRANSLATION,
)
WRIST_CAMERA_LOCAL_ROTATE_XYZ_DEG = parse_vec3_env(
    "Z1_WRIST_CAMERA_ROTATE_XYZ_DEG",
    WRIST_CAMERA_LOCAL_ROTATE_XYZ_DEG,
)
WRIST_CAMERA_FORWARD_VECTOR, WRIST_CAMERA_FORWARD_AXIS = parse_axis_env(
    "Z1_WRIST_CAMERA_FORWARD_AXIS",
    "+X",
)
WRIST_CAMERA_UP_VECTOR, WRIST_CAMERA_UP_AXIS = parse_axis_env(
    "Z1_WRIST_CAMERA_UP_AXIS",
    "+Z",
)

FRONTEND_PATH = Path(os.environ.get("Z1_TELEOP_FRONTEND", str(FRONTEND_PATH)))
if not FRONTEND_PATH.exists():
    local_frontend = Path(__file__).resolve().parents[1] / "z1_teleop_frontend.html"
    if local_frontend.exists():
        FRONTEND_PATH = local_frontend

# Optional: use this if you know the exact end-effector prim name.
# Examples:
#   Z1_WRIST_CAMERA_MOUNT=gripperStator
#   Z1_WRIST_CAMERA_MOUNT=link06
#   Z1_WRIST_CAMERA_MOUNT=gripperMover
PREFERRED_WRIST_CAMERA_MOUNT = os.environ.get("Z1_WRIST_CAMERA_MOUNT", "gripperStator").strip()
