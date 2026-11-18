import numpy as np
import unittest
import inspect
import json
from pathlib import Path
from sensor_msgs.msg import Image

from z1_isaac_bridge.episode_recorder import EpisodeRecorder, image_to_bgr


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "isaac_assets" / "z1").is_dir()
)


def make_image(array, encoding, step=None):
    message = Image()
    message.height = array.shape[0]
    message.width = array.shape[1]
    message.encoding = encoding
    message.step = step or array.shape[1] * array.shape[2]
    message.data = array.tobytes()
    return message


class ImageConversionTest(unittest.TestCase):
    def test_rgb8_is_converted_to_bgr(self):
        rgb = np.array(
            [[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8
        )
        bgr = image_to_bgr(make_image(rgb, "rgb8"))
        np.testing.assert_array_equal(
            bgr,
            np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8),
        )

    def test_row_padding_is_ignored(self):
        padded = np.array(
            [
                [1, 2, 3, 4, 5, 6, 99, 99],
                [7, 8, 9, 10, 11, 12, 99, 99],
            ],
            dtype=np.uint8,
        )
        message = Image()
        message.height = 2
        message.width = 2
        message.encoding = "bgr8"
        message.step = 8
        message.data = padded.tobytes()
        image = image_to_bgr(message)
        np.testing.assert_array_equal(
            image,
            np.array(
                [
                    [[1, 2, 3], [4, 5, 6]],
                    [[7, 8, 9], [10, 11, 12]],
                ],
                dtype=np.uint8,
            ),
        )

    def test_unsupported_encoding_is_rejected(self):
        message = Image()
        message.height = 1
        message.width = 1
        message.encoding = "16UC1"
        message.step = 2
        message.data = bytes(2)
        with self.assertRaisesRegex(ValueError, "unsupported image encoding"):
            image_to_bgr(message)


def test_wrist_rgb_uses_factory_calibrated_16_by_9_profile():
    asset_root = REPO_ROOT / "isaac_assets" / "z1"
    runtime = (asset_root / "scripts" / "run_z1_ros2_scene.py").read_text()
    camera_fix = (
        REPO_ROOT
        / "ros2_ws"
        / "src"
        / "z1_ros2"
        / "z1_examples"
        / "z1_examples"
        / "camera_info_fix.py"
    ).read_text()
    metadata = json.loads(
        (
            asset_root
            / "generated"
            / "smolvla_red_ball_into_box.json"
        ).read_text()
    )
    recorder = inspect.getsource(EpisodeRecorder)

    assert "WRIST_COLOR_WIDTH = 640" in runtime
    assert "WRIST_COLOR_HEIGHT = 360" in runtime
    assert "WRIST_COLOR_FX = 451.606170654297" in runtime
    assert 'self.declare_parameter("color_fx", 451.606170654297)' in camera_fix
    assert metadata["camera"]["color_resolution"] == [640, 360]
    assert metadata["camera"]["depth_resolution"] == [640, 480]
    assert metadata["global_camera"]["resolution"] == [640, 480]
    assert '"wrist": (360, 640, 3)' in recorder
    assert '"global": (480, 640, 3)' in recorder
    assert 'self.expected_image_shapes["vla_wrist"] = (360, 640, 3)' in recorder
    assert "z1_tabletop_raw_dual_wrist" in recorder
    assert "/vla_wrist_camera/color/image_raw" in recorder


if __name__ == "__main__":
    unittest.main()
