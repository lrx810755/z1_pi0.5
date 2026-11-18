import os
import sys
import time
import webbrowser
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from z1_teleop_backend.config import (
    CAMERA_FRAME_INTERVAL_S,
    HAND_EYE_CALIBRATION_PATH,
    HOST,
    MOVEIT_BRIDGE_URL,
    PORT,
    ROBOT_STAGE_PATH,
    URDF_PATH,
    USD_PATH,
)


def validate_paths():
    if not USD_PATH.exists():
        raise FileNotFoundError(f"USD 文件不存在: {USD_PATH}")
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF 文件不存在: {URDF_PATH}")
    if not HAND_EYE_CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"手眼标定文件不存在: {HAND_EYE_CALIBRATION_PATH}")


def main():
    from isaacsim import SimulationApp

    # Isaac/Omniverse modules must be imported only after SimulationApp exists.
    simulation_app = SimulationApp({"headless": False})

    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view

    from z1_teleop_backend.camera import CameraFrameStore, create_wrist_camera, load_hand_eye_calibration
    from z1_teleop_backend.fixed_demo import FixedRedBallDemoController
    from z1_teleop_backend.grasp_controller import RedBallGraspController
    from z1_teleop_backend.http_server import start_http_server
    from z1_teleop_backend.moveit_bridge import MoveItBridge
    from z1_teleop_backend.real_arm_client import RealArmClient
    from z1_teleop_backend.safety import configure_drive_gains, load_urdf_limits
    from z1_teleop_backend.smolvla_client import SmolVLAEvaluatorClient
    from z1_teleop_backend.teleop_state import TeleopState
    from z1_teleop_backend.trajectory_executor import TrajectoryExecutor
    from z1_teleop_backend.usd_utils import cleanup_disabled_red_targets, find_articulation_root, find_prim_path_by_name
    from z1_teleop_backend.vision_red_ball import make_static_scene

    validate_paths()
    joint_limits = load_urdf_limits(URDF_PATH)
    hand_eye_calibration = load_hand_eye_calibration(HAND_EYE_CALIBRATION_PATH)

    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(usd_path=str(USD_PATH), prim_path=ROBOT_STAGE_PATH)
    world.reset()
    set_camera_view(
        eye=np.array([1.25, -1.5, 0.95]),
        target=np.array([0.25, 0.0, 0.28]),
        camera_prim_path="/OmniverseKit_Persp",
    )

    articulation_root = find_articulation_root(world.stage)
    if articulation_root is None:
        raise RuntimeError("未找到 ArticulationRoot，请确认 z1.usd 可被 Isaac 作为机械臂加载。")

    z1 = SingleArticulation(prim_path=articulation_root, name="z1")
    world.scene.add(z1)
    red_ball_manager = make_static_scene(world)
    cleanup_disabled_red_targets(world.stage)
    world.reset()
    z1.initialize()
    if red_ball_manager is not None:
        red_ball_manager.configure_runtime_physics()
    configure_drive_gains(z1)

    wrist_camera, wrist_camera_path, wrist_camera_mount_path = create_wrist_camera(world.stage)
    gripper_body_path = find_prim_path_by_name(world.stage, "link06") or wrist_camera_mount_path
    gripper_attach_paths = [
        find_prim_path_by_name(world.stage, "gripperMover"),
        find_prim_path_by_name(world.stage, "gripperStator"),
        gripper_body_path,
    ]
    gripper_attach_paths = [path for path in gripper_attach_paths if path is not None]

    real_arm = RealArmClient()
    real_arm.start()

    teleop_state = TeleopState(z1, joint_limits, real_arm=real_arm)
    moveit_bridge = MoveItBridge()
    trajectory_executor = TrajectoryExecutor(teleop_state)
    red_ball_grasp = RedBallGraspController(
        teleop=teleop_state,
        camera=wrist_camera,
        calibration=hand_eye_calibration,
        mount_path=wrist_camera_mount_path,
        gripper_path=gripper_body_path,
        gripper_attach_paths=gripper_attach_paths,
        ball_manager=red_ball_manager,
        moveit_bridge=moveit_bridge,
        trajectory_executor=trajectory_executor,
    )
    fixed_demo = FixedRedBallDemoController(
        teleop=teleop_state,
        ball_manager=red_ball_manager,
        attach_body_path=gripper_attach_paths,
    )
    camera_frame_store = CameraFrameStore()
    smolvla_client = SmolVLAEvaluatorClient()
    http_server = start_http_server(
        teleop_state=teleop_state,
        camera_frame_store=camera_frame_store,
        red_ball_manager=red_ball_manager,
        red_ball_grasp=red_ball_grasp,
        trajectory_executor=trajectory_executor,
        fixed_demo=fixed_demo,
        real_arm=real_arm,
        smolvla_client=smolvla_client,
    )
    url = f"http://{HOST}:{PORT}"

    for _ in range(30):
        world.step(render=True)
    camera_frame_store.update(wrist_camera)

    print("========== Z1 Teleop ==========")
    print(f"Loaded robot articulation: {articulation_root}")
    print(f"Wrist camera mount: {wrist_camera_mount_path}")
    print(f"Wrist camera path: {wrist_camera_path}")
    print(f"Gripper attach candidates: {gripper_attach_paths}")
    print("Wrist camera source: embedded USD Camera prim")
    print(f"Frontend camera feed: {wrist_camera_path}")
    print(f"Hand-eye calibration: {HAND_EYE_CALIBRATION_PATH}")
    print(f"MoveIt bridge: {MOVEIT_BRIDGE_URL}")
    print(f"Real Z1 bridge: {real_arm.base_url} enabled={real_arm.enabled}")
    print(f"Real Z1 status: {real_arm.last_status.get('message', '')}")
    print(f"Teleop URL: {url}")
    print("Safety: URDF joint limits, backend velocity limiting, hold, home, and estop are enforced.")
    print("Press Esc in the browser or click 急停 to stop commanded motion.")

    if os.environ.get("Z1_TELEOP_OPEN_BROWSER", "0") == "1":
        webbrowser.open(url)

    try:
        last_time = time.monotonic()
        last_camera_frame_time = 0.0
        while simulation_app.is_running():
            now = time.monotonic()
            dt = max(1e-3, now - last_time)
            last_time = now
            if red_ball_manager is not None:
                red_ball_manager.apply_pending_reset()
            fixed_demo.step()
            trajectory_executor.step()
            red_ball_grasp.step()
            teleop_state.step(dt)
            world.step(render=True)
            if red_ball_manager is not None:
                red_ball_manager.update_attached_pose()
            if now - last_camera_frame_time >= CAMERA_FRAME_INTERVAL_S:
                camera_frame_store.update(wrist_camera)
                last_camera_frame_time = now
    finally:
        http_server.shutdown()
        real_arm.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
