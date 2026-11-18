"""Input and output transforms for the single-active-arm YAM setup."""

import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms

YAM_ACTION_DIM = 7
YAM_GRIPPER_INDEX = 6


def _image_to_hwc_uint8(image: object) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"YAM camera image must be rank 3, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"YAM camera image must have three channels, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("YAM camera image contains non-finite values")
        if array.size and np.nanmax(array) <= 1.0 + 1e-6 and np.nanmin(array) >= -1e-6:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _require_yam_vector(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] != YAM_ACTION_DIM:
        raise ValueError(f"YAM {name} must end in {YAM_ACTION_DIM} values, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"YAM {name} contains non-finite values")
    return array.copy()


def yam_to_pi(values: object, *, name: str) -> np.ndarray:
    """Convert native YAM gripper semantics (0 closed) to pi semantics (0 open)."""
    array = _require_yam_vector(values, name=name)
    gripper = array[..., YAM_GRIPPER_INDEX]
    if np.any((gripper < 0.0) | (gripper > 1.0)):
        raise ValueError(f"YAM {name} gripper must be in [0, 1]")
    array[..., YAM_GRIPPER_INDEX] = 1.0 - array[..., YAM_GRIPPER_INDEX]
    return array


def pi_to_yam(values: object, *, name: str) -> np.ndarray:
    """Convert model-facing pi gripper semantics back to native YAM semantics."""
    array = _require_yam_vector(values, name=name)
    array[..., YAM_GRIPPER_INDEX] = np.clip(1.0 - array[..., YAM_GRIPPER_INDEX], 0.0, 1.0)
    return array


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Map native seven-dimensional YAM observations to OpenPI model inputs."""

    CAMERA_NAMES: ClassVar[tuple[str, ...]] = ("middle", "left", "right")

    def __call__(self, data: dict) -> dict:
        images = data.get("images")
        if not isinstance(images, dict):
            raise ValueError("YAM input is missing the images mapping")
        missing = set(self.CAMERA_NAMES) - set(images)
        extra = set(images) - set(self.CAMERA_NAMES)
        if missing or extra:
            raise ValueError(
                f"YAM cameras must be exactly {self.CAMERA_NAMES}; missing={sorted(missing)}, extra={sorted(extra)}"
            )

        output = {
            "image": {
                "base_0_rgb": _image_to_hwc_uint8(images["middle"]),
                "left_wrist_0_rgb": _image_to_hwc_uint8(images["left"]),
                "right_wrist_0_rgb": _image_to_hwc_uint8(images["right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": yam_to_pi(data["state"], name="state"),
        }
        if "actions" in data:
            output["actions"] = yam_to_pi(data["actions"], name="actions")
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Return native seven-dimensional YAM joint-position commands."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])[..., :YAM_ACTION_DIM]
        return {"actions": pi_to_yam(actions, name="actions")}
