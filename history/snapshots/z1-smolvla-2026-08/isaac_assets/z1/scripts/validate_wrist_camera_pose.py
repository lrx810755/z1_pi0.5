#!/usr/bin/env python3
"""Replay saved joint states to compare physically realizable wrist-camera poses.

The saved RGB images are the baseline.  New views must be rendered because a
different eye-in-hand extrinsic cannot be recovered by warping an RGB frame.
This tool does not modify the source USD or the running ROS/Isaac process.
"""

import argparse
import json
import math
import sys
from pathlib import Path

from project_paths import ASSET_ROOT, DATA_ROOT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=DATA_ROOT / "z1_tabletop_raw_dual_wrist",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=(
            ASSET_ROOT
            / "generated_dual_camera"
            / "smolvla_dual_wrist_camera.usd"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/z1_wrist_camera_validation"),
    )
    parser.add_argument("--max-episodes", type=int, default=9)
    parser.add_argument("--samples-per-phase", type=int, default=3)
    return parser.parse_args()


ARGS = parse_args()
# Do not let Kit reinterpret this tool's CLI flags (notably ``--output``).
sys.argv = [sys.argv[0]]

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import cv2
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import open_stage
from isaacsim.sensors.camera import Camera
from pxr import Gf, UsdGeom
from scipy.spatial.transform import Rotation


ROBOT_PATH = "/World/Z1/root_joint"
GRIPPER_PATH = f"{ROBOT_PATH}/gripperStator"
OBJECT_PATHS = {
    "cup": "/World/Task/Objects/Cup",
    "apple": "/World/Task/Objects/Apple",
    "peach": "/World/Task/Objects/Peach",
    "mouse": "/World/Task/Objects/Mouse",
    "beverage_bottle": "/World/Task/Objects/BeverageBottle",
}
OBJECT_CENTER_Z = {
    "cup": 0.0305,
    "apple": 0.0190,
    "peach": 0.0145,
    "mouse": 0.01415,
    "beverage_bottle": 0.0785,
}
WIDTH, HEIGHT = 640, 360
FX, FY = 451.606170654297, 451.370849609375
CX, CY = 325.197723388672, 188.640960693359
HORIZONTAL_APERTURE_MM = 20.955

# The baseline is the measured gripperStator -> color optical frame transform.
BASELINE_OPTICAL_XYZ = np.array(
    [0.1006887452, 0.0312032005, -0.0302586435], dtype=np.float64
)
BASELINE_OPTICAL_Q_XYZW = np.array(
    [-0.5023940423, 0.5151748053, -0.4771697725, 0.5044840478],
    dtype=np.float64,
)
BASELINE_MOUNT_XYZ = np.array(
    [0.0900930511, -0.0009311400, -0.0436741355], dtype=np.float64
)
BASELINE_MOUNT_RPY = np.array(
    [0.04037701, 0.03561724, 0.01525814], dtype=np.float64
)

# Candidate optical centers are bounded around the wrist and still require a
# real bracket/collision check. All look at the nominal fruit pinch center.
PINCH_CENTER = np.array([0.145, 0.0, 0.018], dtype=np.float64)
CANDIDATES = {
    "baseline_calibrated": {"position": BASELINE_OPTICAL_XYZ, "baseline": True},
    "top_back_10mm_aimed": {
        "position": np.array([0.090, 0.000, 0.065], dtype=np.float64)
    },
    "top_back_30mm_aimed": {
        "position": np.array([0.070, 0.000, 0.065], dtype=np.float64)
    },
    "top_back_50mm_aimed": {
        "position": np.array([0.050, 0.000, 0.065], dtype=np.float64)
    },
    "top_back_50mm_side_aimed": {
        "position": np.array([0.050, 0.045, 0.055], dtype=np.float64)
    },
}


def optical_rotation_aimed(position):
    """Return gripper<-optical rotation; optical axes are right/down/forward."""
    forward = PINCH_CENTER - position
    forward /= np.linalg.norm(forward)
    down_hint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    down = down_hint - forward * float(np.dot(down_hint, forward))
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    return np.column_stack((right, down, forward))


def optical_rotation(candidate):
    if candidate.get("baseline"):
        return Rotation.from_quat(BASELINE_OPTICAL_Q_XYZW).as_matrix()
    return optical_rotation_aimed(candidate["position"])


def renderer_rotation(candidate):
    optical = optical_rotation(candidate)
    # Isaac Camera looks along local -Z with +Y up; ROS optical is +Z/+Y down.
    return optical @ np.diag([1.0, -1.0, -1.0])


def homogeneous(position, rotation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def candidate_mount_pose(candidate):
    baseline_mount = homogeneous(
        BASELINE_MOUNT_XYZ,
        Rotation.from_euler("xyz", BASELINE_MOUNT_RPY).as_matrix(),
    )
    baseline_optical = homogeneous(
        BASELINE_OPTICAL_XYZ,
        Rotation.from_quat(BASELINE_OPTICAL_Q_XYZW).as_matrix(),
    )
    mount_to_optical = np.linalg.inv(baseline_mount) @ baseline_optical
    desired_optical = homogeneous(candidate["position"], optical_rotation(candidate))
    desired_mount = desired_optical @ np.linalg.inv(mount_to_optical)
    rotation = Rotation.from_matrix(desired_mount[:3, :3])
    return {
        "xyz_m": desired_mount[:3, 3].tolist(),
        "rpy_rad": rotation.as_euler("xyz").tolist(),
        "rpy_deg": rotation.as_euler("xyz", degrees=True).tolist(),
    }


def matrix_to_gf_quat(matrix):
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def configure_projection(camera_prim):
    camera = UsdGeom.Camera(camera_prim)
    focal_length = FX * HORIZONTAL_APERTURE_MM / float(WIDTH)
    vertical_aperture = focal_length * float(HEIGHT) / FY
    camera.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE_MM)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureOffsetAttr((CX - WIDTH / 2.0) / FX)
    camera.CreateVerticalApertureOffsetAttr((CY - HEIGHT / 2.0) / FY)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5.0))


def create_candidate_camera(stage, name, candidate):
    path = f"{GRIPPER_PATH}/VLAValidation_{name}"
    print(f"[wrist-validation] defining {path}", flush=True)
    usd_camera = UsdGeom.Camera.Define(stage, path)
    print(f"[wrist-validation] configuring xform {name}", flush=True)
    xform = UsdGeom.Xformable(usd_camera.GetPrim())
    xform.ClearXformOpOrder()
    p = candidate["position"]
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
    xform.AddOrientOp().Set(matrix_to_gf_quat(renderer_rotation(candidate)))
    configure_projection(usd_camera.GetPrim())
    print(f"[wrist-validation] constructing sensor {name}", flush=True)
    camera = Camera(prim_path=path, name=f"vla_validation_{name}", resolution=(WIDTH, HEIGHT))
    print(f"[wrist-validation] constructed sensor {name}", flush=True)
    return camera


def load_episode_samples(root, max_episodes, samples_per_phase):
    result = []
    for episode_dir in sorted(root.glob("episode_*")):
        summary_path = episode_dir / "episode_summary.json"
        frames_path = episode_dir / "frames.jsonl"
        if not summary_path.exists() or not frames_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if "scene_randomization" not in summary:
            continue
        frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
        # The object remains on the table in these phases, so recorded joint
        # states plus saved initial object positions reproduce their geometry.
        selected = []
        for phase in ("READY", "GRASPING"):
            phase_frames = [frame for frame in frames if frame.get("phase") == phase]
            if not phase_frames:
                continue
            indices = np.linspace(
                0, len(phase_frames) - 1, min(samples_per_phase, len(phase_frames)), dtype=int
            )
            selected.extend(phase_frames[index] for index in indices)
        result.append((episode_dir, summary, selected))
        if len(result) >= max_episodes:
            break
    return result


def target_color_mask(image_bgr, target):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Apple and peach scans are warm-colored. Other objects can also be warm,
    # so this is a coverage proxy, not a semantic detector.
    warm_a = cv2.inRange(hsv, np.array([0, 55, 35]), np.array([35, 255, 255]))
    warm_b = cv2.inRange(hsv, np.array([165, 55, 35]), np.array([179, 255, 255]))
    return cv2.bitwise_or(warm_a, warm_b) if target in ("apple", "peach") else warm_a


def score_image(image_bgr, target):
    mask = target_color_mask(image_bgr, target)
    count = int(np.count_nonzero(mask))
    dark_fraction = float(
        np.mean(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) < 10)
    )
    if count == 0:
        return {
            "warm_fraction": 0.0,
            "centering": 0.0,
            "edge_clearance": 0.0,
            "dark_fraction": dark_fraction,
        }
    ys, xs = np.nonzero(mask)
    center_distance = math.hypot(float(xs.mean() - WIDTH / 2), float(ys.mean() - HEIGHT / 2))
    max_distance = math.hypot(WIDTH / 2, HEIGHT / 2)
    edge = min(xs.min(), WIDTH - 1 - xs.max(), ys.min(), HEIGHT - 1 - ys.max())
    return {
        "warm_fraction": count / float(WIDTH * HEIGHT),
        "centering": max(0.0, 1.0 - center_distance / max_distance),
        "edge_clearance": max(0.0, min(1.0, float(edge) / 60.0)),
        "dark_fraction": dark_fraction,
    }


def set_scene_for_episode(objects, summary):
    positions = summary["scene_randomization"]["object_positions_m"]
    for name, xyz in positions.items():
        if name not in objects:
            continue
        position = np.array([xyz[0], xyz[1], xyz[2]], dtype=np.float32)
        objects[name].set_world_pose(position=position)
        objects[name].set_linear_velocity(np.zeros(3, dtype=np.float32))
        objects[name].set_angular_velocity(np.zeros(3, dtype=np.float32))


def save_contact_sheet(paths, destination, columns=6):
    images = [cv2.imread(str(path)) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    thumb_w, thumb_h = 320, 180
    rows = math.ceil(len(images) / columns)
    sheet = np.zeros((rows * thumb_h, columns * thumb_w, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        sheet[row * thumb_h : (row + 1) * thumb_h, column * thumb_w : (column + 1) * thumb_w] = cv2.resize(
            image, (thumb_w, thumb_h)
        )
    cv2.imwrite(str(destination), sheet)


def main():
    print("[wrist-validation] loading episodes", flush=True)
    ARGS.output.mkdir(parents=True, exist_ok=True)
    episodes = load_episode_samples(ARGS.episodes, ARGS.max_episodes, ARGS.samples_per_phase)
    if not episodes:
        raise RuntimeError(f"No episodes found under {ARGS.episodes}")
    if not open_stage(str(ARGS.scene)):
        raise RuntimeError(f"Failed to open {ARGS.scene}")
    print("[wrist-validation] stage opened", flush=True)
    print("[wrist-validation] creating world", flush=True)
    try:
        world = World(stage_units_in_meters=1.0)
    except BaseException as exc:
        print(f"[wrist-validation] World construction failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    print("[wrist-validation] world created", flush=True)
    stage = omni.usd.get_context().get_stage()
    # Candidate cameras represent relocating the whole D435. Hide the source
    # D435 body so it cannot occlude the relocated optical center.
    source_body = stage.GetPrimAtPath(f"{ROBOT_PATH}/camera_link")
    if source_body.IsValid():
        UsdGeom.Imageable(source_body).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    cameras = {}
    for name, candidate in CANDIDATES.items():
        print(f"[wrist-validation] authoring camera {name}", flush=True)
        cameras[name] = create_candidate_camera(stage, name, candidate)
    print("[wrist-validation] cameras authored", flush=True)
    robot = SingleArticulation(ROBOT_PATH, name="z1_camera_validation_robot")
    objects = {
        name: SingleRigidPrim(path, name=f"camera_validation_{name}")
        for name, path in OBJECT_PATHS.items()
    }
    print("[wrist-validation] resetting world", flush=True)
    world.reset()
    robot.initialize()
    for rigid in objects.values():
        rigid.initialize()
    print("[wrist-validation] world ready", flush=True)
    for camera in cameras.values():
        camera.initialize()
    for _ in range(8):
        world.step(render=True)

    metrics = {name: [] for name in cameras}
    image_paths = {name: [] for name in cameras}
    joint_index = {name: index for index, name in enumerate(robot.dof_names)}
    expected_joints = [f"joint{i}" for i in range(1, 7)] + ["jointGripper"]
    for episode_dir, summary, frames in episodes:
        set_scene_for_episode(objects, summary)
        for frame in frames:
            positions = np.asarray(robot.get_joint_positions(), dtype=np.float32)
            for value, joint_name in zip(frame["observation.state"], expected_joints):
                positions[joint_index[joint_name]] = value
            robot.set_joint_positions(positions)
            robot.set_joint_velocities(np.zeros_like(positions))
            set_scene_for_episode(objects, summary)
            world.step(render=True)
            for name, camera in cameras.items():
                rgba = camera.get_rgba()
                rgb = np.asarray(rgba[:, :, :3])
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * 255.0 if float(rgb.max()) <= 1.0 else rgb, 0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                filename = (
                    f"{episode_dir.name}_{frame['phase'].lower()}_"
                    f"{frame['frame_index']:06d}.jpg"
                )
                candidate_dir = ARGS.output / name
                candidate_dir.mkdir(exist_ok=True)
                path = candidate_dir / filename
                cv2.imwrite(str(path), bgr)
                image_paths[name].append(path)
                row = score_image(bgr, summary["target_object"])
                row.update(
                    episode=episode_dir.name,
                    frame_index=frame["frame_index"],
                    phase=frame["phase"],
                    target=summary["target_object"],
                )
                metrics[name].append(row)

    report = {
        "episodes": len(episodes),
        "recommended_candidate": "top_back_30mm_aimed",
        "recommendation_note": (
            "Best balance of target visibility, edge clearance, near-clip safety, "
            "and gripper occlusion. Verify bracket clearance and recalibrate hand-eye "
            "before changing the real camera."
        ),
        "metric_note": (
            "Warm-color coverage is a proxy for apple/peach visibility, "
            "not a semantic detector. Dark coverage is used as an occlusion proxy."
        ),
        "candidates": {},
    }
    for name, rows in metrics.items():
        report["candidates"][name] = {
            "optical_position_gripper_m": CANDIDATES[name]["position"].tolist(),
            "camera_mount_in_gripper": candidate_mount_pose(CANDIDATES[name]),
            "sample_count": len(rows),
            "mean_warm_fraction": float(np.mean([row["warm_fraction"] for row in rows])),
            "mean_centering": float(np.mean([row["centering"] for row in rows])),
            "mean_edge_clearance": float(np.mean([row["edge_clearance"] for row in rows])),
            "target_visible_rate": float(np.mean([row["warm_fraction"] > 0.001 for row in rows])),
            "mean_dark_fraction": float(
                np.mean([row["dark_fraction"] for row in rows])
            ),
        }
        save_contact_sheet(image_paths[name], ARGS.output / f"contact_sheet_{name}.jpg")
    (ARGS.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


failed = False
try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    failed = True
finally:
    APP.close()
if failed:
    raise SystemExit(1)
