import unittest
import importlib.util

from menlo_runner.perception import detect_color_blobs
from menlo_runner.navigation import angle_error_deg


class NavigationMathTest(unittest.TestCase):
    def test_angle_error_straight_ahead(self):
        self.assertAlmostEqual(angle_error_deg((0.0, 0.0), 0.0, (1.0, 0.0)), 0.0)

    def test_angle_error_left(self):
        self.assertAlmostEqual(angle_error_deg((0.0, 0.0), 0.0, (0.0, 1.0)), 90.0)

    def test_angle_error_wraps_to_shortest_turn(self):
        self.assertAlmostEqual(angle_error_deg((0.0, 0.0), 170.0, (1.0, 0.0)), -170.0)


class PerceptionTest(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("cv2") is None, "opencv-python is not installed")
    def test_detect_color_blobs_returns_empty_for_blank_image(self):
        import cv2
        import numpy as np

        blank = np.zeros((120, 160, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", blank)
        self.assertTrue(ok)
        self.assertEqual(detect_color_blobs(encoded.tobytes()), [])


if __name__ == "__main__":
    unittest.main()

