import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from z1_examples.target_detector import DetectorConfig, TargetDetector


class FakeBox:
    def __init__(self, cls_id, confidence, xyxy):
        self.cls = np.array([cls_id], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class FakeModel:
    names = {0: "person", 32: "sports ball"}

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, _image, **_kwargs):
        return [SimpleNamespace(boxes=self._boxes, names=self.names)]


class TargetDetectorTest(unittest.TestCase):
    def test_pixel_detector_finds_red_center(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(image, (80, 55), 18, (0, 0, 255), -1)
        result = TargetDetector(
            DetectorConfig(mode="pixel", min_area_px=100)
        ).detect(image)
        self.assertTrue(result.found)
        self.assertAlmostEqual(result.u, 80.0, delta=1.0)
        self.assertAlmostEqual(result.v, 55.0, delta=1.0)
        self.assertIsNotNone(result.mask)
        self.assertIsNotNone(result.contour)

    def test_pixel_detector_rejects_small_target(self):
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.circle(image, (40, 40), 3, (0, 0, 255), -1)
        result = TargetDetector(
            DetectorConfig(mode="pixel", min_area_px=100)
        ).detect(image)
        self.assertFalse(result.found)
        self.assertEqual(result.reason, "no_pixel_target")

    def test_yolo_filters_class_and_prefers_confidence(self):
        boxes = [
            FakeBox(0, 0.99, [0, 0, 100, 100]),
            FakeBox(32, 0.55, [10, 10, 30, 30]),
            FakeBox(32, 0.85, [40, 20, 80, 60]),
        ]
        detector = TargetDetector(
            DetectorConfig(mode="yolo", target_class="sports ball", min_area_px=10),
            model=FakeModel(boxes),
        )
        result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertTrue(result.found)
        self.assertEqual(result.label, "sports ball")
        self.assertAlmostEqual(result.u, 60.0)
        self.assertAlmostEqual(result.v, 40.0)

    def test_yolo_reports_missing_target_class(self):
        detector = TargetDetector(
            DetectorConfig(mode="yolo", target_class="cup", min_area_px=10),
            model=FakeModel([FakeBox(32, 0.9, [10, 10, 40, 40])]),
        )
        result = detector.detect(np.zeros((60, 60, 3), dtype=np.uint8))
        self.assertFalse(result.found)
        self.assertEqual(result.reason, "yolo_class_not_found:cup")

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            DetectorConfig(mode="unknown")
        with self.assertRaises(ValueError):
            DetectorConfig(confidence=1.5)


if __name__ == "__main__":
    unittest.main()
