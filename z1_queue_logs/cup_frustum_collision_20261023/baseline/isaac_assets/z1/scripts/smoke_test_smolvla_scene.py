#!/usr/bin/env python3
"""Runtime smoke test for the generated SmolVLA Isaac Sim task scene."""

import os
from pathlib import Path

from isaacsim import SimulationApp
from z1_paths import asset_root


APP = SimulationApp({"headless": True})

import cv2
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import open_stage
from isaacsim.sensors.camera import Camera


SCENE_PATH = Path(
    os.environ.get(
        "Z1_SMOKE_SCENE",
        str(asset_root() / "generated" / "smolvla_red_ball_into_box.usd"),
    )
)
PREVIEW_DIR = Path(os.environ.get("Z1_SMOKE_PREVIEW_DIR", str(SCENE_PATH.parent)))
PREVIEW_PATH = PREVIEW_DIR / "wrist_color_preview.png"
VLA_PREVIEW_PATH = PREVIEW_DIR / "vla_wrist_color_preview.png"
COLOR_CAMERA_PATH = "/World/Z1/root_joint/camera_color_optical_frame/ColorCamera"
VLA_COLOR_CAMERA_PATH = (
    "/World/Z1/root_joint/vla_camera_color_optical_frame/VLAColorCamera"
)


def main() -> None:
    try:
        if not SCENE_PATH.exists():
            raise FileNotFoundError(SCENE_PATH)
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        if not open_stage(str(SCENE_PATH)):
            raise RuntimeError(f"Failed to open scene: {SCENE_PATH}")

        world = World(stage_units_in_meters=1.0)
        camera = Camera(
            prim_path=COLOR_CAMERA_PATH,
            name="smolvla_wrist_color",
            resolution=(640, 360),
        )
        vla_camera = Camera(
            prim_path=VLA_COLOR_CAMERA_PATH,
            name="smolvla_vla_wrist_color",
            resolution=(640, 360),
        )
        world.reset()
        camera.initialize()
        vla_camera.initialize()
        for _ in range(45):
            world.step(render=True)

        previews = {}
        for name, sensor, camera_path, path in (
            ("wrist", camera, COLOR_CAMERA_PATH, PREVIEW_PATH),
            (
                "vla_wrist",
                vla_camera,
                VLA_COLOR_CAMERA_PATH,
                VLA_PREVIEW_PATH,
            ),
        ):
            rgba = sensor.get_rgba()
            if rgba is None or rgba.shape[:2] != (360, 640):
                raise RuntimeError(
                    f"Invalid {name} camera frame: "
                    f"{None if rgba is None else rgba.shape}"
                )
            rgb = np.asarray(rgba[:, :, :3])
            if rgb.dtype != np.uint8:
                rgb = np.clip(
                    rgb * 255.0 if float(rgb.max()) <= 1.0 else rgb,
                    0,
                    255,
                ).astype(np.uint8)
            if float(np.std(rgb)) < 1.0:
                raise RuntimeError(
                    f"{name} camera frame is nearly uniform; setup is invalid"
                )
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to write preview: {path}")
            previews[name] = {
                "camera": camera_path,
                "frame_shape": list(rgba.shape),
                "rgb_mean": float(np.mean(rgb)),
                "rgb_std": float(np.std(rgb)),
                "preview": str(path),
            }

        print(
            {
                "status": "ok",
                "scene": str(SCENE_PATH),
                "cameras": previews,
            },
            flush=True,
        )
    finally:
        APP.close()


main()
