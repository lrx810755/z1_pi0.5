"""Shared pixel and Ultralytics YOLO target detection for Z1 examples."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np


def resolve_yolo_model_path(model_path: str) -> str:
    """Resolve a bare model name to a local .pt file."""
    if os.path.isabs(model_path) or Path(model_path).exists():
        return model_path
    env_dir = os.environ.get("Z1_MODELS_DIR")
    if env_dir:
        candidate = Path(env_dir) / model_path
        if candidate.exists():
            return str(candidate)
    try:
        from ament_index_python.packages import get_package_share_path

        candidate = Path(get_package_share_path("z1_examples")) / "models" / model_path
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return model_path


@dataclass(frozen=True)
class DetectorConfig:
    mode: str = "yolo"
    model_path: str = "yolo11n.pt"
    target_class: str = "bottle"
    confidence: float = 0.35
    min_area_px: float = 200.0
    min_saturation: int = 80
    min_value: int = 50
    hue_high1: int = 10
    hue_low2: int = 170
    device: Optional[str] = None

    def __post_init__(self):
        if self.mode not in ("pixel", "yolo"):
            raise ValueError(f"Unsupported detector mode: {self.mode!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1")
        if self.min_area_px < 0.0:
            raise ValueError("Minimum target area cannot be negative")


@dataclass
class DetectionResult:
    found: bool
    reason: str = ""
    u: float = 0.0
    v: float = 0.0
    area: float = 0.0
    bbox: Optional[Tuple[float, float, float, float]] = None
    label: str = ""
    confidence: Optional[float] = None
    mask: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None


class TargetDetector:
    """Detector with lazy, cached YOLO model loading."""

    def __init__(self, config: DetectorConfig, model: Any = None):
        self.config = config
        self._model = model
        self._loaded_model_path = config.model_path if model is not None else None

    def update_config(self, config: DetectorConfig):
        if config.model_path != self.config.model_path:
            self._model = None
            self._loaded_model_path = None
        self.config = config

    def class_names(self):
        model = self._load_yolo_model()
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        return [str(name) for name in names]

    def detect(self, image: np.ndarray) -> DetectionResult:
        if image is None or image.ndim != 3 or image.shape[2] < 3:
            return DetectionResult(False, reason="invalid_image")
        if self.config.mode == "yolo":
            return self._detect_yolo(image)
        return self._detect_pixel(image)

    def _detect_pixel(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cfg = self.config
        lower1 = np.array([0, cfg.min_saturation, cfg.min_value], dtype=np.uint8)
        upper1 = np.array([cfg.hue_high1, 255, 255], dtype=np.uint8)
        lower2 = np.array([cfg.hue_low2, cfg.min_saturation, cfg.min_value], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= cfg.min_area_px]
        if not contours:
            return DetectionResult(False, reason="no_pixel_target", mask=mask)
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            return DetectionResult(False, reason="invalid_contour", mask=mask)
        x, y, width, height = cv2.boundingRect(contour)
        return DetectionResult(
            True,
            u=float(moments["m10"] / moments["m00"]),
            v=float(moments["m01"] / moments["m00"]),
            area=float(cv2.contourArea(contour)),
            bbox=(float(x), float(y), float(x + width - 1), float(y + height - 1)),
            label="red target",
            mask=mask,
            contour=contour,
        )

    def _load_yolo_model(self):
        if self._model is not None and self._loaded_model_path == self.config.model_path:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO 依赖未安装；请在 ROS2 Python 环境安装 requirements-yolo.txt"
            ) from exc
        model_path = resolve_yolo_model_path(self.config.model_path)
        if not Path(model_path).exists():
            raise RuntimeError(
                f"找不到 YOLO 模型 {self.config.model_path!r}；"
                "请直接传绝对路径，或把模型放到 Z1_MODELS_DIR 指向的目录"
            )
        try:
            self._model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(f"无法加载 YOLO 模型 {self.config.model_path!r}: {exc}") from exc
        self._loaded_model_path = self.config.model_path
        return self._model

    def _detect_yolo(self, image):
        model = self._load_yolo_model()
        kwargs = {"conf": self.config.confidence, "verbose": False}
        if self.config.device:
            kwargs["device"] = self.config.device
        try:
            results = model.predict(image, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"YOLO 推理失败: {exc}") from exc
        if not results:
            return DetectionResult(False, reason="yolo_no_results")
        result = results[0]
        names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return DetectionResult(False, reason="yolo_no_objects")

        target = self.config.target_class.strip().casefold()
        candidates = []
        available_labels = set()
        for box in boxes:
            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, cls_id) if isinstance(names, dict) else names[cls_id])
            available_labels.add(label)
            if target and label.casefold() != target:
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            area = max((x2 - x1) * (y2 - y1), 0.0)
            if area < self.config.min_area_px:
                continue
            candidates.append((confidence, area, x1, y1, x2, y2, label))
        if not candidates:
            reason = "yolo_target_not_found"
            if target and target not in {label.casefold() for label in available_labels}:
                reason = f"yolo_class_not_found:{self.config.target_class}"
            return DetectionResult(False, reason=reason)
        confidence, area, x1, y1, x2, y2, label = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        return DetectionResult(
            True,
            u=0.5 * (x1 + x2),
            v=0.5 * (y1 + y2),
            area=area,
            bbox=(x1, y1, x2, y2),
            label=label,
            confidence=confidence,
        )


def detector_config_from_args(args) -> DetectorConfig:
    mode = getattr(args, "detector", "yolo")
    if getattr(args, "yolo", False):
        mode = "yolo"
    return DetectorConfig(
        mode=mode,
        model_path=getattr(args, "yolo_model", "yolo11n.pt"),
        target_class=getattr(args, "yolo_class", "bottle"),
        confidence=float(getattr(args, "yolo_conf", 0.35)),
        min_area_px=float(args.min_area_px),
        min_saturation=int(args.min_saturation),
        min_value=int(args.min_value),
        hue_high1=int(args.hue_high1),
        hue_low2=int(args.hue_low2),
        device=getattr(args, "yolo_device", None) or None,
    )


def add_detector_arguments(parser):
    parser.add_argument("--detector", choices=["yolo", "pixel"], default="yolo")
    parser.add_argument("--yolo", action="store_true", help="Deprecated alias for --detector yolo")
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    parser.add_argument("--yolo-class", default="bottle")
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--yolo-device", default="", help="Ultralytics device, for example cpu or 0; empty means auto")
